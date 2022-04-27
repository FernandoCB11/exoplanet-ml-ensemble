from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import pandas as pd
import joblib
import multiprocessing
import numpy
from scipy.stats import entropy
from astropy.timeseries import TimeSeries
import os
import math
from sklearn.metrics import mean_squared_error
from astropy.table import vstack
from pandas import DataFrame


class SIDRA:
    """
    Class which implements the SIDRA Random Forest model from
    MISLIS et al. (2016).

    For this implementation, we've used SKLearn's RandomForestClassifier class
    following all the definitions of Mislis' original work, including the
    feature engineering.

    For hiperparameters which were not discussed on the original work,
    we kept RandomForestClassifier's defaults.

    Attributes
    ----------
    model: RandomForestClassifier
        RandomForestClassifier model trained.
    name: str
        A Name label for this class. Will be used as a standard for dumps and
        result file naming.
    sidra_input_file: str
        Path to the file where the preprocessed lightcurves will be stored.
    kepler_dir: str
        Directory where the original lightcurve files are.
    """

    def __init__(self, sidra_input_file: str, kepler_dir: str):
        """
        Parameters
        ----------
        sidra_input_file: str
            Path to the file where the preprocessed lightcurves will be stored.
        kepler_dir: str
            Directory where the original lightcurve files are.
        """
        self.model = None
        self.name = "SIDRA"
        self.sidra_input_file = sidra_input_file  # "../data/sidra_input/inputs_original.txt"  # noqa E501
        self.kepler_dir = kepler_dir

    def train(self, sample):
        """
        Trains the SIDRA Classifier.

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
        RuntimeError
            If the preprocessed lightcurves file is not found.
        """
        if not os.path.isfile(self.sidra_input_file):
            raise RuntimeError("Preprocessed lightcurves file not found. Run the preprocess method first.")  # noqa E501

        preprocessed_lightcurves = pd.read_csv(
            self.sidra_input_file,
            delimiter=",",
            header=0
        )
        preprocessed_lightcurves.drop_duplicates(inplace=True)

        x_train = pd.merge(
            sample,
            preprocessed_lightcurves,
            how="inner",
            on='kepid'
        )

        # Creates the classifier with the informed hiperparameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=7
        )

        model.fit(
            x_train[['skewness', 'kurtosis', 'autocorrelation', 'entropy']],
            x_train[['av_training_set']]
        )

        self.model = model

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
            If the preprocessed lightcurves file is not found.
        """
        if self.model is None:
            raise RuntimeError("Model not trained, run the train method first.")  # noqa E501

        if not os.path.isfile(self.sidra_input_file):
            raise RuntimeError("Preprocessed lightcurves file not found. Run the preprocess method first.")  # noqa E501

        preprocessed_lightcurves = pd.read_csv(
            self.sidra_input_file,
            delimiter=",",
            header=0
        )
        preprocessed_lightcurves.drop_duplicates(inplace=True)

        X = pd.merge(
            sample,
            preprocessed_lightcurves,
            how="inner",
            on='kepid'
        )

        results = self.model.predict(
            X[['skewness', 'kurtosis', 'autocorrelation', 'entropy']]
        )
        X['predicted'] = results

        return X[['rowid', 'predicted']]

    def dump(self, target_file: str = None):
        """
        Dumps the trained model to a file.

        Parameters
        ----------
        target_file: str
            Name of the file that is going the be created for the model dump.
            If its None, assumes `../models/SIDRA_MODEL_YYYYMMDDHHMiSS` where
            YYYY is the current year, MM is the current month, DD is the
            current day, HH is the current hour, Mi is the current minute and
            SS is the current second.
        """
        if target_file is None:
            target_file = "../models/SIDRA_MODEL_" + datetime.now().strftime("%Y%m%d%H%M%S")  # noqa E501
        joblib.dump(self, target_file)

    def pre_process(self, tce_table: DataFrame, num_parallel_process: int = 3):
        """
        Executes the lightcurve preprocessing and creates, following the
        equations of the original, the lightcurve features:
            skewness
            kurtosis
            autocorrelation (autocorrelation integral)
            entropy
        This process should be executed only once per set of lightcurves and
        the set does not need to be split in any way.

        Parameters
        ----------
        tce_table: DataFrame
            DataFrame
            Data frame containing the TCE information. It must contain the
            following columns:
                tce_period
                tce_duration
                tce_time0bk
                av_training_set
                kepid
        num_parallel_process: int
            Number tce to be processed in parallel.

        Raises
        ------
        RuntimeError
            If the tce_table is none.
        """
        if os.path.exists(self.sidra_input_file):
            os.remove(self.sidra_input_file)

        if tce_table is None:
            raise RuntimeError("TCE table not informed.")

        with multiprocessing.Pool(num_parallel_process) as pool:
            [
                pool.apply(
                    self._pre_process,
                    args=(
                        row['kepid'],
                        row['av_training_set'],
                        self.sidra_input_file
                    )
                )
                for index, row in tce_table.iterrows()
            ]

    def _pre_process(self, kepid: str, kepler_dir: str, result_file: str):
        """
        Executes the lightcurve preprocessing for one TCE.

        Each feature is defined in the equations of MISLIS et al. (2016):

        Skewness (S) eq. 7
        Kurtosis (K) eq. 8
        Autocorrelation Integral (A) eq. 9
        Entropy (E) eqs. 10 and 11

        Parameters
        ----------
        kepid: str
            Id of the KOI associated with the TCE, it is the id used to
            retrieve the lightcurve files.
        kepler_dir: str
            Directory where the original lightcurve files are. It is received
            as a parameter and not a class attribute to enable multiprocessing.
        result_file: str
            Name of the resulting file where the features of the preprocessed
            lightcurve will be writen to. It is received as a parameter and
            not a class attribute to enable multiprocessing.
        """
        padded_kepid = str(kepid).zfill(9)

        try:
            files = os.listdir(os.path.join(kepler_dir, padded_kepid))
            timeseries = None
            if len(files) == 0:
                raise RuntimeError(f"Lightcurve files not found for KOI {padded_kepid}.")  # noqa E501

            for file in files:
                # Loads the lightcurve as a time series into an aux variable
                timeseries_aux = TimeSeries.read(
                    os.path.join(kepler_dir, padded_kepid, file),
                    format='kepler.fits'
                )

                if timeseries is None:
                    # If it is the first file use it as a start for the
                    # timeseries
                    timeseries = timeseries_aux
                else:
                    # Else stack the read timeseries with the previous ones.
                    timeseries = vstack(timeseries, timeseries_aux)

            # clears the lightcurve of zeros and NaNs
            lightcurve = numpy.array(timeseries['pdcsap_flux'])
            lightcurve = lightcurve[numpy.where(lightcurve > 0)]

            mean = numpy.mean(lightcurve)
            std_dev = numpy.std(lightcurve)
            n = lightcurve.size
            rmse = math.sqrt(mean_squared_error(lightcurve, [0 for _ in lightcurve]))  # noqa E501

            skewness = 0
            kurtosis = 0
            autocorrelation = 0
            entropy_lc = 0

            pd_series = pd.Series(lightcurve)
            counts = pd_series.value_counts()
            entropy_lc = entropy(counts)

            for i in range(n - 1):
                skewness += math.pow((lightcurve[i] - mean), 3) / math.pow(std_dev, 3)  # noqa E501
                kurtosis += math.pow((lightcurve[i] - mean), 4) / math.pow(std_dev, 4)  # noqa E501
                delta = 0

                # For the delta we invert the indexes
                # i = delay and j = array index
                for j in range(i):
                    if (j + i) > (n - 1):
                        break
                    delta += (lightcurve[j] - mean) * (lightcurve[j+i] - mean)

                autocorrelation += 1/((n - i) * math.pow(rmse, 2)) * delta

            autocorrelation = abs(autocorrelation)  # Get the absolute value
            skewness = skewness/n  # Divide by the number of measures

            file_output = open(result_file, "a")

            file_output.write(f"{kepid},{str(skewness)},{str(kurtosis)},{str(autocorrelation)},{str(entropy_lc)}\n")  # noqa E501
            file_output.close()

        except Exception as e:
            print(str(e))
            raise e
