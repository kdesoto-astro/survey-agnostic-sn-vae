import os

import numpy as np

from snapi.query_agents import TNSQueryAgent, ALeRCEQueryAgent
from snapi.transient import Transient, Photometry   

def canonicalize(spec_class):
    """Canonicalize a specific spectroscopic class."""
    if spec_class in ["SN Ia", "SN Ia-91T-like", "SN Ia-91bg-like", "SN Ia-CSM"]:
        return "SN Ia"
    if spec_class in ["SN II", "SN IIL", "SN IIP"]:
        return "SN II"
    if spec_class in ["SN Ib", "SN Ic", "SN Ic-BL", "SN Ib/c", "SN Ib-Ca-rich", "SN Ic-Ca-rich"]:
        return "SN Ibc"
    if spec_class in ["SN IIn", "SLSN-II"]:
        return "SN IIn"
    if spec_class in ["SLSN-I"]:
        return "SLSN-I"
    return spec_class

def generate_ztf_transients(
    save_dir,
):
    """Generate ZTF transients for the VAE."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)

    tns_agent = TNSQueryAgent()
    alerce_agent = ALeRCEQueryAgent()

    all_names = tns_agent.retrieve_all_names() # only spectroscopically classified

    for i, n in enumerate(all_names):
        if i % 1000 == 0:
            print(f"Processed {i} of {len(all_names)} objects.")
        # if file exists already, just skip
        #if os.path.exists(os.path.join(save_dir, n+".hdf5")):
        #    print(f"SKIPPED {n}: File already exists")
        #    continue
        transient = Transient(iid=n)
        qr_tns, success = tns_agent.query_transient(transient, local=True) # we dont want spectra
        if not success:
            print("SKIPPED {n}: TNS query failed")
            continue
        for result in qr_tns:
            transient.ingest_query_info(result.to_dict())
        qr_alerce, success = alerce_agent.query_transient(transient)
        if not success:
            print(f"SKIPPED {n}: ALeRCE query failed")
            continue
        for result in qr_alerce:
            transient.ingest_query_info(result.to_dict())

        # quality cuts
        photometry = transient.photometry.filter_by_instrument("ZTF")

        new_lcs = set()
        for lc in photometry.light_curves:
            lc.calibrate_fluxes(23.90)
            high_snr_detections = lc.detections[lc.detections['mag_unc'] <= (5 / 8. / np.log(10))] # SNR >= 4

            # number of high-SNR detections cut
            if len(high_snr_detections) < 5:
                continue

            # variability cut
            if (
                np.max(high_snr_detections['mag']) - np.min(high_snr_detections['mag'])
             ) < 3 * np.mean(high_snr_detections['mag_unc']):
                continue

            # second variability cut
            if np.std(high_snr_detections['mag']) < np.mean(high_snr_detections['mag_unc']):
                continue
            
            new_lcs.add(lc)
        
        if len(new_lcs) == 0:
            continue

        # convert to absolute magnitudes
        cal_photometry = Photometry(new_lcs)
        absolute_photometry = cal_photometry.absolute(transient.redshift)
        # correct for extinction
        corrected_photometry = absolute_photometry.correct_extinction(
            coordinates=transient.coordinates
        )
        transient.photometry = corrected_photometry
        transient.spec_class = canonicalize(transient.spec_class)
        transient.save(os.path.join(save_dir, n+".hdf5"))