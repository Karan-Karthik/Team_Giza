# utils.py

US_REGIONS = {
    'Northeast': [
        'ME',
        'NH',
        'VT',
        'MA',
        'RI',
        'CT',
        'NY',
        'NJ',
        'PA'],
    'Midwest': [
        'OH',
        'MI',
        'IN',
        'IL',
        'WI',
        'MN',
        'IA',
        'MO',
        'ND',
        'SD',
        'NE',
        'KS'],
    'South': [
        'DE',
        'MD',
        'DC',
        'VA',
        'WV',
        'NC',
        'SC',
        'GA',
        'FL',
        'KY',
        'TN',
        'AL',
        'MS',
        'AR',
        'LA',
        'OK',
        'TX'],
    'West': [
        'MT',
        'ID',
        'WY',
        'CO',
        'NM',
        'AZ',
        'UT',
        'NV',
        'WA',
        'OR',
        'CA',
        'AK',
        'HI']}


def get_region_for_state(state):
    """Map state to US region."""
    for region, states in US_REGIONS.items():
        if state in states:
            return region
    return 'Other'


def get_states_for_selection(selections):
    """Get list of states for the selected states/regions."""
    states = []
    for selection in selections:
        if selection in US_REGIONS:
            states.extend(US_REGIONS[selection])
        else:
            states.append(selection)
    return list(set(states))  # Remove duplicates
