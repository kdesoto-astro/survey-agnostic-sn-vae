import warnings
import collections
import itertools
import jax

from survey_agnostic_sn_vae.archival.data_adapter_utils import get_data_adapter, ddistribution
from survey_agnostic_sn_vae.archival import tree

def _distribute_data(data):
    distribution = ddistribution()
    
    if distribution is not None:

        def distribute_single_value(d):
            #layout = distribution.get_data_layout(d.shape)
            raise NotImplementedError()

            #return jax_distribution_lib.distribute_data_input(d, layout)

        return tree.map_structure(distribute_single_value, data)
    else:
        return tree.map_structure(jax.device_put, data)
        
class EpochIterator:
    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps_per_epoch=None,
        shuffle=False,
        class_weight=None,
        steps_per_execution=1,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution
        if steps_per_epoch:
            self._current_iterator = None
            self._insufficient_data = False
        self.data_adapter = get_data_adapter(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
        )
        self._num_batches = self.data_adapter.num_batches

    def _get_iterator(self):
        return self.data_adapter.get_numpy_iterator()

    def enumerate_epoch(self):
        buffer = []
        if self.steps_per_epoch:
            if self._current_iterator is None:
                self._current_iterator = iter(self._get_iterator())
                self._insufficient_data = False

            for step in range(self.steps_per_epoch):
                if self._insufficient_data:
                    break

                try:
                    data = next(self._current_iterator)
                    buffer.append(data)
                    if len(buffer) == self.steps_per_execution:
                        yield step - len(buffer) + 1, buffer
                        buffer = []
                except (StopIteration,):
                    warnings.warn(
                        "Your input ran out of data; interrupting epoch. "
                        "Make sure that your dataset or generator can generate "
                        "at least `steps_per_epoch * epochs` batches. "
                        "You may need to use the `.repeat()` "
                        "function when building your dataset.",
                        stacklevel=2,
                    )
                    self._current_iterator = None
                    self._insufficient_data = True
            if buffer:
                yield step - len(buffer) + 1, buffer
        else:
            for step, data in enumerate(self._get_iterator()):
                buffer.append(data)
                if len(buffer) == self.steps_per_execution:
                    yield step - len(buffer) + 1, buffer
                    buffer = []
            if buffer:
                yield step - len(buffer) + 1, buffer
            if not self._num_batches:
                # Infer the number of batches returned by the data_adapter.
                # Assumed static.
                self._num_batches = step + 1
        self.data_adapter.on_epoch_end()

    @property
    def num_batches(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # Either copied from the data_adapter, or
        # inferred at the end of an iteration.
        return self._num_batches
        
class JAXEpochIterator(EpochIterator):
    def _get_iterator(self):
        return self._prefetch_numpy_iterator(
            self.data_adapter.get_jax_iterator()
        )

    def _prefetch_numpy_iterator(self, numpy_iterator):
        """Shard and prefetch batches on device.

        Most of the implementation has been borrowed from
        `flax.jax_utils.prefetch_to_device`

        This utility takes an iterator and returns a new iterator which fills an
        on device prefetch buffer. Eager prefetching can improve the performance
        of training loops significantly by overlapping compute and data
        transfer.
        """
        queue = collections.deque()

        # If you're training on GPUs, 2 is generally the best choice because
        # this guarantees that you can overlap a training step on GPU with a
        # data prefetch step on CPU.
        def enqueue(n=2):
            for data in itertools.islice(numpy_iterator, n):
                queue.append(_distribute_data(data))

        enqueue(n=2)  # TODO: should we make `n` configurable?
        while queue:
            yield queue.popleft()
            enqueue(1)