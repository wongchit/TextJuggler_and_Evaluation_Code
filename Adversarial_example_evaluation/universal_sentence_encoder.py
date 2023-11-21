"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import importlib
import types


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    This allows them to only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Lazy module loader cannot find module named `{self.__name__}`. "
                f"This might be because TextAttack does not automatically install some optional dependencies. "
                f"Please run `pip install {self.__name__}` to install the package."
            ) from e
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class UniversalSentenceEncoder:
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="cosine", **kwargs):
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

        self.model = hub.load(tfhub_url)

    def encode(self, sentences):
        return self.model(sentences).numpy()
