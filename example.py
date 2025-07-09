from DeepFMKit.core import DeepFitFramework

def main():

    print("Program start!")
    print("Loading raw data from \'/test/raw_data.txt\'...")

    dff = DeepFitFramework(raw_file='./test/raw_data.txt', raw_labels=['ch1'])

    print(f"Added DeepRawObject to \'raws\' dictionary: {dff.raws}")
    print("Fitting...")

    dff.fit('ch1', method='nls', n=20, parallel=False)

    print(f"Added DeepFitObject to \'fits\' dictionary: {dff.fits}")
    print("Done.")

if __name__ == '__main__':
    main()