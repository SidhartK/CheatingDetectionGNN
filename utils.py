def cheating_probability(row1, row2, w1=0.2, w2=0.5, w3=0.3, t_thresh=10, verbose=False):
    """
    Calculates the probability of two users cheating based on answer similarity and timing similarity.

    Parameters
    ----------
    row1 : pd.Series
        First user's data.
    row2 : pd.Series
        Second user's data.
    w1 : float, optional
        Weight for correct/incorrect answer matching (CIM), default is 0.2.
    w2 : float, optional
        Weight for matching incorrect answers (MIA), default is 0.5.
    w3 : float, optional
        Weight for timing similarity (TS), default is 0.3.
    t_thresh : int, optional
        Time threshold in seconds for considering two times as similar, default is 10.
    verbose : bool, optional
        Whether to print intermediate results, default is False.

    Returns
    -------
    float
        Probability of cheating between 0 and 1.
    """
    # Correct/Incorrect Answer Matching (CIM)
    # Calculate the number of correct/incorrect answer matches
    num_data_points = 10
    cim_matches = sum([row1[f'C{i}'] == row2[f'C{i}'] for i in range(1, num_data_points + 1)])
    # Calculate the probability of CIM
    p_cim = cim_matches ** 2 / num_data_points ** 2
    
    # Matching Incorrect Answers (MIA)
    # Calculate the number of matching incorrect answers
    mia_matches = 0
    for i in range(1, num_data_points + 1):
        if row1[f'C{i}'] == 0 and row2[f'C{i}'] == 0 and row1[f'A{i}'] == row2[f'A{i}']:
            # Calculate the rarity weight for each answer
            rarity_weight = 1 - row1[f'A{i}_freq']
            mia_matches += rarity_weight
    # Calculate the probability of MIA
    p_mia = 1 - np.exp(-0.8 * (mia_matches ** 2))
    
    # Timing Similarity (TS)
    # Calculate the number of timing matches
    ts_matches = 0
    for i in range(1, num_data_points + 1):
        # Check if either time is blank
        if not pd.isna(row1[f'T{i}']) and not pd.isna(row2[f'T{i}']):
            # Calculate the absolute difference in time in seconds
            time_diff = abs((pd.to_datetime(row1[f'T{i}']) - pd.to_datetime(row2[f'T{i}'])).total_seconds())
            if time_diff <= t_thresh:
                ts_matches += 1
    # Calculate the probability of TS
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

