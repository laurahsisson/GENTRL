import numpy as np

class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def display_dict(self):
        return {str(key): ": {:4.4};".format(
                np.mean(values)
            ) for key, values in self.stats.items() if len(values) > 0}

    def print(self):
        for key, values in self.stats.items():
            if len(values) > 0:
                print(str(key) + ": {:4.4};".format(
                    np.mean(values)
                ), end='')

        print()