
def Distance_normalizer(distance):

    if len(distance) == 1:
        Normal_distance = distance/distance
    else:
        min_val = torch.min(distance)
        max_val = torch.max(distance)

        # Normalize the distances between 0 and 1
        Normal_distance = (distance - min_val) / (max_val - min_val)
    return Normal_distance