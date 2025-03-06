from functions.plot_stacked_raster_and_psth import plot_stacked_raster_and_psth
import os

# Define figures_dir here so it is available in this module
figures_dir = os.path.join("reports", "figures")
os.makedirs(figures_dir, exist_ok=True)

def plot_group_figures(group_name, neurons, water_events, sugar_events, cta_time):
    if cta_time is not None:
        water_pre = water_events[water_events < cta_time]
        sugar_pre = sugar_events[sugar_events < cta_time]
        water_post = water_events[water_events >= (cta_time + 3 * 3600)]
        sugar_post = sugar_events[sugar_events >= (cta_time + 3 * 3600)]
    else:
        # If no CTA time, treat all events as pre
        water_pre = water_events
        sugar_pre = sugar_events
        water_post = []
        sugar_post = []

    # Figure 1: Pre-CTA
    pre_title = f"{group_name} Group - Pre-CTA (Water & Sugar)"
    pre_save = os.path.join(figures_dir, f"{group_name}_pre_CTA_stacked.png")
    plot_stacked_raster_and_psth(neurons, water_pre, sugar_pre,
                                 window=(-0.5, 2.0), bin_width=0.05,
                                 figure_title=pre_title, save_path=pre_save)
    
    # Figure 2: Post-CTA
    post_title = f"{group_name} Group - Post-CTA (Water & Sugar)"
    post_save = os.path.join(figures_dir, f"{group_name}_post_CTA_stacked.png")
    plot_stacked_raster_and_psth(neurons, water_post, sugar_post,
                                 window=(-0.5, 2.0), bin_width=0.05,
                                 figure_title=post_title, save_path=post_save)