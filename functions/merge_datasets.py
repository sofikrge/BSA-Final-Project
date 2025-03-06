import numpy as np

def merge_datasets(group_ds):
    """
    Merge the neurons and event times for a set of datasets.
    Returns:
      combined_neurons, combined_water_events, combined_sugar_events, common_cta_time
    For CTA injection time, we use the CTA time from the first dataset in the group.
    """
    combined_neurons = []
    combined_water   = []
    combined_sugar   = []
    cta_times = []
    
    for ds in group_ds.values():
        combined_neurons.extend(ds["neurons"])
        data = ds["data"]
        combined_water.extend(data["event_times"].get("water", []))
        combined_sugar.extend(data["event_times"].get("sugar", []))
        cta = data.get("CTA injection time", None)
        if cta is not None:
            cta_times.append(cta)
    
    common_cta = cta_times[0] if cta_times else None
    return (
        combined_neurons, 
        np.array(combined_water), 
        np.array(combined_sugar), 
        common_cta
    )