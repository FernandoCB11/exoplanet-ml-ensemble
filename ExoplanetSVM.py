from sklearn.svm import SVC
from datetime import datetime
import numpy as np
import joblib
from AstroNet_preprocessing import AstroNetPreprocessing


class ExoplanetSVM:
    """
    Attributes
    ----------
    data_dir: str
        Base directory where the lightcurve files are stored.
    num_worker_processes: int
        Number of worker processess to be spawned in the parallel processing of
        the input files.
    cache_dir: str
        Cache directory for preprocessed lightcurves.
    """

    def __init__(self, base_dir: str, data_dir: str, cache_dir: str):  # noqa E501
        """
        Parameters
        ----------
        base_dir: str
            Base directory where the tfrecord and model dirs will be created.
        model_dir: str
            Directory where the files for the model should be created.
        cache_dir: str
            Cache directory for preprocessed lightcurves.
        """
        self.model = None
        self.data_dir = data_dir
        self.name = "EXOPLANETSVM"
        self.preprocessing = AstroNetPreprocessing(
            base_dir=base_dir,
            data_dir=self.data_dir,
            cache_dir=cache_dir
        )

    def train(self, sample):
        """
        Trains the ExoplanetSVM Classifier.

        Parameters
        ----------
        sample: DataFrame
            Pandas Dataframe with the TCE information of the training set.
            Must have the following columns:
                tce_period
                tce_duration
                tce_time0bk
                av_training_set
                kepid

        Raises
        ------
        """
        x_train = self.preprocess(sample)

        model = SVC(C=1.0, kernel='rbf', gamma='scale', tol=0.001)

        model.fit(x_train, sample['av_training_set'])

        self.model = model

    def preprocess(self, sample):
        """
        Execute the preprocessing for ExoplanetSVM.
        This preprocessing is the retrieval of the lightcurve's global view,
        from AstroNet preprocessing steps.

        Parameters
        ----------
        sample: DataFrame
            Pandas Dataframe with the TCE information of the training set.
            Must have the following columns:
                tce_period
                tce_duration
                tce_time0bk
                av_training_set
                kepid

        Returns
        -------
        X: list
            List with the global view of the lightcurves, in order they are
            in the sample input.
        """
        X = []
        for index, row in sample.iterrows():
            try:
                ex = self.preprocessing._process_tce(
                    row,
                    kepler_dir=self.data_dir
                )
                global_view = np.array(
                    (ex.features.feature["global_view"]).float_list.value
                )

                X.append(global_view)
            except Exception as e:
                print(str(e))
                X.append(np.zeros(shape=(2001,), dtype=np.float32))
        return X

    def predict(self, sample):
        """
        Uses the trained classifier to predict the class of a set of records.

        Parameters
        ----------
        sample: DataFrame
            Pandas Dataframe with the TCE information.
            Must have the following columns:
                tce_period
                tce_duration
                tce_time0bk
                kepid

        Returns
        -------
        result: DataFrame
            Data frame with the predicted CLASS for each TCE.
            The columns are `rowid` and `predicted` with the predicted class.
            The possible values are 1 for the `PC` class and 0 for the `AFP`
            class.

        Raises
        ------
        RuntimeError
            If the model is not trained.
        """
        if self.model is None:
            raise RuntimeError("Model not trained, run the train method first.")  # noqa E501

        X = self.preprocess(sample)

        results = self.model.predict(X)
        ret = sample[['rowid']]
        ret['predicted'] = results

        return ret

    def dump(self, file=None):
        """
        Dumps the trained model to a file.

        Parameters
        ----------
        target_file: str
            Name of the file that is going the be created for the model dump.
            If its None, assumes `../models/SVM_MODEL_YYYYMMDDHHMiSS` where
            YYYY is the current year, MM is the current month, DD is the
            current day, HH is the current hour, Mi is the current minute and
            SS is the current second.
        """
        if file is None:
            file = "../models/SVM_MODEL_" + datetime.now().strftime("%Y%m%d%H%M%S")  # noqa E501
        joblib.dump(self, file)
