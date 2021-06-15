import os
import pickle as pkl


def save_pickle(obj, directory, filename):
    with open(os.path.join(directory, filename), "wb") as f:
        pkl.dump(obj, f)


def load_pickle(directory, filename):
    with open(os.path.join(directory, filename), "rb") as f:
        obj = pkl.load(f)
    return obj


class ResultSaver:
    """
    Convenience class to persist experimental results right away
    """

    def __init__(self, dir):
        """
        Creates a ResultSaver that maintains an internal dict to save results and persist them later
        on. Also, the instance ensures that all model are persisted in a separate subdirectory.
        :param dir: location where results should be persisted
        """

        self.dir = dir
        self.default_file = "results.pkl"
        self.model_dir = os.path.join(dir, "models")
        self.results = {}

        os.makedirs(self.model_dir,
                    exist_ok=True)  # creates also the root directory in case it doesn't exists

    def save(self, key, new_obj, parents=[], force_persist=False):
        """
        Saves an object in the internal dictionary. The dict hierarchy can be traversed via
        the parents argument.
        :param key: new key
        :param new_obj: new object
        :param parents: parent keys
        :param force_persist: indicates whether the internal dir is persisted immediately
        :return self
        """
        obj = self.results
        for parent in parents:
            obj = obj[parent]
        obj[key] = new_obj

        if force_persist:
            self.persist()

        return self

    def persist(self):
        """
        persists internal results
        :return:
        """
        save_pickle(self.results, self.dir, self.default_file)

    def persist_model(self, model, name):
        """
        Persists the model in ./<root>/models subdirectory
        :param model: model to persist
        :param name: how to name the file
        """
        file = os.path.join(self.model_dir, name) + ".h5"
        model.save(file)
