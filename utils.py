def cheating_probability(row1, row2, t_thresh=10, verbose = False):
    """
    Calculates the probability of two users cheating based on answer similarity and timing similarity.

    Parameters:
    row1 (pd.Series): First user's data.
    row2 (pd.Series): Second user's data.
    k1 (float): Weight for correct/incorrect answer matching.
    k2 (float): Weight for matching incorrect answers.
    k3 (float): Weight for timing similarity.
    t_thresh (int): Time threshold in seconds for considering two times as similar.
    verbose (bool): Whether to print intermediate results.

    Returns:
    float: Probability of cheating between 0 and 1.
    """
    w1, w2, w3 = 0.2, 0.5, 0.3
    # Correct/Incorrect Answer Matching (CIM)
    num_data_points = 10
    cim_matches = sum([row1[f'C{i}'] == row2[f'C{i}'] for i in range(1, num_data_points + 1)])
    p_cim = cim_matches**2/num_data_points**2
    
    # Matching Incorrect Answers (MIA)
    mia_matches = 0
    for i in range(1, num_data_points + 1):
        if row1[f'C{i}'] == 0 and row2[f'C{i}'] == 0 and row1[f'A{i}'] == row2[f'A{i}']:
            rarity_weight = 1 - row1[f'A{i}_freq']  # Higher weight for rarer answers
            mia_matches += rarity_weight
    p_mia = 1 - np.exp(-0.8 * (mia_matches ** 2))
    
    # Timing Similarity (TS)
    ts_matches = 0
    for i in range(1, num_data_points + 1):
        # Check if either time is blank
        if not pd.isna(row1[f'T{i}']) and not pd.isna(row2[f'T{i}']):
            # Updated time format to YYYY/MM/DD HH:MM:SS
            if abs((pd.to_datetime(row1[f'T{i}']) - pd.to_datetime(row2[f'T{i}'])).total_seconds()) <= t_thresh:
                ts_matches += 1
    p_ts = 1 - np.exp(-0.6 * ts_matches)
    
    # Combine probabilities
    cheating_prob = p_cim * w1 + p_mia * w2 + p_ts * w3
    if verbose:
        print("User1:", row1['id'], "User2:", row2['id'])
        print("CIM matches:", cim_matches)
        print("p_cim:", p_cim)
        print("MIA matches:", mia_matches)
        print("p_mia:", p_mia)
        print("TS matches:", ts_matches)
        print("p_ts:", p_ts)
        print("Cheating probability:", cheating_prob)
    return cheating_prob
