from ..transforms.hrir import DecibelTransform, DomainTransform, SelectValueRangeTransform
from sklearn.base import BaseEstimator, TransformerMixin


class SKLearnAdapter(BaseEstimator, TransformerMixin):

    def __init__(self, transform_type, *args, **kwargs):
        self.transform_type
        self.args = args
        self.kwargs = kwargs


    def fit(self, X, _y=None):
        self._transform = self.transform_type(*self.args, **self.kwargs)
        return self


    def transform(self, X, _y=None):
        return self._transform(X)


    def inverse_transform(self, X, _y=None):
        return self._transform.inverse(X)


class Flatten(BaseEstimator, TransformerMixin):

    def fit(self, X, _y=None):
        self._shape = X.shape[1:]
        return self


    def transform(self, X, _y=None):
        return X.reshape(len(X), -1)


    def inverse_transform(self, X, _y=None):
        return X.reshape(-1, *self._shape)


class DcRemoval(BaseEstimator, TransformerMixin):

    def __init__(self, time_domain=False):
        self.time_domain = time_domain


    @staticmethod
    def _subtract_mean(X):
        return X - X.mean(axis=-1, keepdims=True)


    @staticmethod
    def _drop_first_bin(X):
        return X[..., 1:]


    def fit(self, X, _y=None):
        if self.time_domain:
            self._transform = self._subtract_mean
        else:
            self._transform = self._drop_first_bin
        return self


    def transform(self, X, _y=None):
        return self._transform(X)


class DecibelTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, _y=None):
        self._transform = DecibelTransform()
        return self


    def transform(self, X, _y=None):
        return self._transform(X)


class FrequencyRegion(BaseEstimator, TransformerMixin):

    def __init__(self, frequencies, lower_freq, upper_freq):
        self.frequencies = frequencies
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq


    def fit(self, X, _y=None):
        self._transform = SelectValueRangeTransform(self.frequencies, self.lower_freq, self.upper_freq)
        return self


    def transform(self, X, _y=None):
        return self._transform(X)


    @property
    def frequencies_(self):
        return self.frequencies[self._transform._selection]


class DomainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, domain):
        self.domain = domain


    def fit(self, X, _y=None):
        self._transform = DomainTransform(self.domain, X.dtype)
        return self


    def transform(self, X, _y=None):
        return self._transform(X)
