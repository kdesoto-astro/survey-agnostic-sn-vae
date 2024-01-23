import os, glob

def generate_superraenn_lc_file(
    transient_dir,
    save_path
):
    """Generate lcs.npz file for use by SuperRAENN.
    """
    all_transient_files = glob.glob(
        os.path.join(transient_dir, "*.pickle")
    )
    all_transients = [
        Transient.load(t) for t in all_transient_files
    ]
    
    # This needs to be redone when retrained
    # TODO: Need to change this whenever you retrain...
    filt_dict = {'g': 0, 'r': 1, 'i': 2, 'z': 3}
    wvs = np.asarray([5460, 6800, 7450, 8700])

    my_lcs = []

    # Update the LC objects with info from the metatable
    for transient in all_transients:
        for lc in transient.lightcurves:
            params = self.model_params
        
            t, f, ferr, b = lc.get_arrays()
            sr_lc = Lightcurve(
                name=lc.obj_id,
                times=t,
                fluxes=f,
                flux_errs=ferr,
                filters=b,
                obj_type=transient.model_type
            )
            
            sr_lc.add_LC_info(
                zpt=params['zeropoint'],
                mwebv=0.0,
                redshift=params['redshift'],
                lim_mag=args.lm,
                obj_type=obj_types[i]
            )
            
            sr_lc.wavelengths = lc.survey.wavel
            sr_lc.get_abs_mags()
            sr_lc.sort_lc()
            pmjd = my_lc.find_peak(
                lc.find_max_flux()[0]
            )
            sr_lc.shift_lc(pmjd)
            sr_lc.correct_time_dilation()
            sr_lc.filter_names_to_numbers(filt_dict)
            sr_lc.cut_lc()
            sr_lc.make_dense_LC(len(lc.bands))
            sr_lcs.append(my_lc)