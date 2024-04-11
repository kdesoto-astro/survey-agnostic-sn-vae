import pandas as pd



def find_start_of_data(fn):
    """Helper function for YSE fn import.
    """
    with open(fn, 'r') as f:
        for i, row in enumerate(f):
            if row.strip() == '':
                return i + 5
            
def import_single_yse_lc(fn):
    """Imports single YSE SNANA file.
    """
    header = find_start_of_data(fn)
    
    df = pd.read_csv(
        fn, header=header,
        on_bad_lines='skip',
        sep='\s+'
    )
    
    df = df.drop(columns='VARLIST:')
    
    # filter into ZTF and YSE LCs
    ztf_lc = df[(df.FLT == 'X') | (df.FLT == 'Y')]
    yse_lc = df[~df.index.isin(ztf_lc.index)]
    
    for i, lc in enumerate([ztf_lc, yse_lc]):
        f, ferr = convert_mag_to_flux(
            , merr, DEFAULT_ZPT
        )

        sr_lc = LightCurve(
            name=lc.obj_id,
            times=t,
            fluxes=f,
            flux_errs=ferr,
            filters=b,
        )
    
    
    
if __name__ == "__main__":
    test_fn = '../../../data/yse_dr1_zenodo_snr_geq_4/2019pmd.snana.dat'
    import_single_yse_lc(test_fn)