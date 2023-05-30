

def create_team_b(alphas, tau, no_arms):
    """Create a list of TeamMember objects. The verbosity of each member is
    determined by alphas (list containing alpha for each member) and the team
    exploration-exploitation strategy is determined by tau (float). The number
    of arms of the MAB to be played must also be specified using no_arms (int)"""
    team = []
    for alpha in alphas:
        team.append(TeamMember(alpha, no_arms, tau))
    return team


class TeamMember:
    """ A TeamMember has attributes alpha (verbosity), tau (exploration-exploitation
    strategy), and no_arms (number of arms of the MAB that the member will play). A
    TeamMember can convert their belief to a choice probability and update their belief
    by incorporating the obtained reward associated with a choice """

    def __init__(self, alpha, no_arms, tau):
        """ Initialize team member alpha (float), tau (float), belief (all options
        are equal when initializing) and choice_probabilities (all options are
        equal when initializing) """
        self.alpha = alpha
        self.belief = [0.5]*no_arms
        self.tau = tau

    def update_beliefs(self, choice, reward):
        """ Update beliefs by incorporating the obtained reward associated with
        the member's choice in accordance with exponential smoothing"""
        self.belief[choice] = (self.alpha*reward) + ((1-self.alpha)*self.belief[choice])
