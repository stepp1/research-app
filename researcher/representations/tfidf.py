"""
Obtained and adapted from the Texthero library: https://github.com/jbesomi/texthero

Preferred to not install the library due to:
    - large amount of unnecesary deps
    - not so flexible methods of clustering, decomposition, etc.
    - lack of methods for building representations (or only traditional ones) 

The texthero.tifidf() method is used to build a tfidf representation of a text-based Pandas Series.
"""

import warnings

import pandas as pd
from preprocessing import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

_not_tokenized_warning_message = (
    "It seems like the given Pandas Series s is not tokenized. This"
    " function will tokenize it automatically using hero.tokenize(s)"
    " first. You should consider tokenizing it yourself first with"
    " hero.tokenize(s) in the future."
)


def tfidf(
    s: pd.Series,
    max_features=1000,
    min_df=1,
    max_df=1.0,
) -> pd.DataFrame:
    """
    Represent a text-based Pandas Series using TF-IDF.
    Rows of the returned DataFrame represent documents whereas columns are
    terms. The value in the cell document-term is the tfidf-value of the
    term in this document. The output is sparse.
    *Term Frequency - Inverse Document Frequency (TF-IDF)* is a formula to
    calculate the _relative importance_ of the words in a document, taking
    into account the words' occurences in other documents. It consists of
    two parts:
    The *term frequency (tf)* tells us how frequently a term is present
    in a document, so tf(document d, term t) = number of times t appears
    in d.
    The *inverse document frequency (idf)* measures how _important_ or
    _characteristic_ a term is among the whole corpus (i.e. among all
    documents). Thus, idf(term t) = log((1 + number of documents) /
    (1 + number of documents where t is present)) + 1.
    Finally, tf-idf(document d, term t) = tf(d, t) * idf(t).
    Different from the `sklearn-implementation of tfidf
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_
    extraction.text.TfidfVectorizer.html>`, this function does *not*
    normalize the output in any way, so the result is exactly what you
    get applying the formula described above.
    The input Series should already be tokenized. If not, it will
    be tokenized before tfidf is calculated.
    Parameters
    ----------
    s : Pandas Series (tokenized)
    max_features : int, optional, default=None
        If not None, only the max_features most frequent tokens are used.
    min_df : float in range [0.0, 1.0] or int, optional, default=1
        When building the vocabulary ignore terms that have a document
        frequency (number of documents they appear in) strictly
        lower than the given threshold.
        If float, the parameter represents a proportion of documents,
        integer absolute counts.
    max_df : float in range [0.0, 1.0] or int, default=1.0
        Ignore terms that have a document frequency (number of documents they
        appear in) frequency strictly higher than the given threshold.
        This arguments basically permits to remove corpus-specific stop
        words. If float, the parameter represents a proportion of documents,
        integer absolute counts.
    Examples
    --------
    >>> import texthero as hero
    >>> import pandas as pd
    >>> s = pd.Series(["Hi Bye", "Test Bye Bye"]).pipe(hero.tokenize)
    >>> hero.tfidf(s) # doctest: +SKIP
        Bye        Hi      Test
    0   1.0  1.405465  0.000000
    1   2.0  0.000000  1.405465
    See Also
    --------
    `TF-IDF on Wikipedia <https://en.wikipedia.org/wiki/Tf-idf>`_
    TODO add tutorial link
    """

    # Check if input is tokenized. Else, print warning and tokenize.
    if not isinstance(s.iloc[0], list):
        warnings.warn(_not_tokenized_warning_message, DeprecationWarning)
        s = tokenize(s)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        use_idf=True,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        tokenizer=lambda x: x,
        token_pattern=None,
        preprocessor=lambda x: x,
        norm=None,  # Disable l1/l2 normalization.
    )

    tfidf_vectors_csr = tfidf.fit_transform(s)

    return pd.DataFrame.sparse.from_spmatrix(
        tfidf_vectors_csr, s.index, tfidf.get_feature_names_out()
    )
