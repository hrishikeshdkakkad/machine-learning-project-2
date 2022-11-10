class Contract:
    def __init__(self):
        self.bestWeights = []
        self.user_weight_mapping = {}
        self.gov_tkn_mapping = {}
        self.arjunTokens = 100

    def issue_governance_token(self, address):
        self.gov_tkn_mapping[address] += 1

    def getAllTrans(self):
        #
        return self.user_weight_mapping



    def send(self, weight, address):
        # exposed to clients
        self.user_weight_mapping[address] = weight

        # poll governance tkn users
