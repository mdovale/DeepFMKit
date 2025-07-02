from DeepFMKit.core import DeepFitFramework

def main():

    print("Program start!")

    dff = DeepFitFramework()

    label = "dynamic"
    dff.new_sim(label)
    dff.sims[label].m = 6.0
    dff.sims[label].f_mod = 1000
    dff.sims[label].f_samp = int(200e3)
    dff.sims[label].f_n = 1e6
    dff.sims[label].arml_mod_f = 1.0
    dff.sims[label].arml_mod_amp = 1e-9
    dff.sims[label].arml_mod_n = 1e-12
    dff.sims[label].fit_n = 10

    dff.simulate(label, n_seconds=3, simulate="dynamic", ref_channel=False)

    dff.fit(label)

    print("Done.")

if __name__ == '__main__':
    main()