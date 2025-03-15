# BSA Final Project Denise Jaeschke & Sofia Karageorgiou

If you want to run it for yourself, here is what you need to do:
    
    - Make sure you have all dependencies installed. 
    
    - Create the data folder and within it the raw folder.
    
    - Place the raw recordings within the data/raw folder.
    
    - Run the finalbsa.py file You can track the progress in the terminal and watch the code create the figures within the reports/figures folder.

Note: If you are not simply copying this repo, please make sure to replicate the structure seen below to make sure the figures will be rendered 
in the intended folder and the script has access to the raw data. The important folders and files you should double-check are marked with an emoji.

Thank you and enjoy!

## Project Organization

```
├── README.md
├── finalbsa.py        <- 🌱 This is the main file that you need to run          
├── data               <- 🍄 Please create this folder
│   ├── processed      <- The final, data sets after exclusions
│   └── raw            <- 🍄 Create this folder and place the raw data we were provided with for the assignment here please 
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         bsa_final_project_dj_&_sk and configuration for 
├── reports            <- General folder for the outputs of our analysis
│   └── figures        <- You will find all figure folders within this one. You do not have to create all the folders within this one, they will be generated automatically 
│       └── Correlograms
│       └── CV_FF
│       └── Firing_Rates
│       └── ProcessedCorrelograms
│       └── PSTH_Raster
│       └── PSTH_TwoByTwo
│       └── Survivor_Function
│       └── Survivor_Summary
│       └── TIH
│
└── functions          <- Contains all of our helper functions we call in our main file
    ├── __init__.py    <- Init file for inheritance
    ├── analyze_firing_rates.py     <- Processe and analyze firing rates across different datasets
    ├── apply_manual_fusion.py      <- Applies manual neuron modifications (fusion and deletion) based on explicitly assigned neuron indices in the main file
    ├── correlogram.py              <- Calculates the correlogram employing a difference-based approach
    ├── cv_fano.py                  <- Calculates and plots the CV and FF metrics
    ├── isi_tih.py                  <- Calculates the ISIs and plots their distribution in a TIH, if filtering is enabled it also filters out biologically impossible spikes
    ├── load_dataset.py             <- Helper function to load specific keys of the dataset dictionary
    ├── plot_correlogram_matrix.py  <- Plots all correlograms for a dataset in a single matrix and aligns them according to their neuron number
    ├── plot_survivor.py            <- Calculates and plots the survivor functions
    ├── psth_rasterplot.py          <- Generates and saves a single figure showing both pre-CTA and post-CTA raster plots and PSTH overlays side by side, mooving-average smoothing
    ├── psth_twobytwo.py            <- Creates 2x2 raster-style histogram plots per neuron + creates a group-level summary, has baseline-correction and mooving-average smoothing
```

--------
