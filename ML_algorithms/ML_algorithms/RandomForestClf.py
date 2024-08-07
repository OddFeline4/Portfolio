class MyForestClf:
    def __init__(self,n_estimators: int =10, max_features: float = 0.5,
                 max_samples:float = 0.5, random_state: int = 42, max_depth:int = 5,
                 min_samples_split:int = 2, max_leafs: int=20, bins:int = 16,criterion='entropy'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion
        self.random_state = random_state




    def __str__(self):
        return 'MyForestClf class: ' + f'{", ".join([str(i[0])+"="+str(i[1]) for i in self.__dict__.items()])}'
