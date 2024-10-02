"""A replay buffer that efficiently stores and can sample whole paths."""
import collections

import numpy as np


class PathBufferEx:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, pixel_shape, sample_goals: bool = False, discount: float = 0.99, put_remaining_mass_on_last_state: bool = False, recurrent: bool = False, sampled_seq_len: int = None):
        self._capacity = capacity_in_transitions
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}

        if pixel_shape is not None:
            self._pixel_dim = np.prod(pixel_shape)
        else:
            self._pixel_dim = None
        self._pixel_keys = ['obs', 'next_obs']

        self.sample_goals = sample_goals
        self.discount = discount
        self.idx_to_horizon = { i: (self._capacity - 1) for i in range(self._capacity) }
        self.idx_to_start = { i: 0 for i in range(self._capacity) }
        self.put_remaining_mass_on_last_state = put_remaining_mass_on_last_state

        # for recurrent versions
        self.recurrent = recurrent
        self._valid_starts = np.zeros(self._capacity, dtype=np.float32)
        self._sampled_seq_len = sampled_seq_len
        self._top = 0

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            if self._pixel_dim is not None and key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                if pixel_key not in self._buffer:
                    self._buffer[pixel_key] = np.random.randint(0, 255, (self._capacity, self._pixel_dim), dtype=np.uint8)  # For memory preallocation
                    self._buffer[state_key] = np.zeros((self._capacity, array.shape[1] - self._pixel_dim), dtype=array.dtype)
                self._buffer[pixel_key][first_seg.start:first_seg.stop] = array[:len(first_seg), :self._pixel_dim]
                self._buffer[state_key][first_seg.start:first_seg.stop] = array[:len(first_seg), self._pixel_dim:]
                self._buffer[pixel_key][second_seg.start:second_seg.stop] = array[len(first_seg):, :self._pixel_dim]
                self._buffer[state_key][second_seg.start:second_seg.stop] = array[len(first_seg):, self._pixel_dim:]
            else:
                buf_arr = self._get_or_allocate_key(key, array)
                buf_arr[first_seg.start:first_seg.stop] = array[:len(first_seg)]
                buf_arr[second_seg.start:second_seg.stop] = array[len(first_seg):]

                # for recurrent version
                if self.recurrent:
                    indices = list(
                        np.arange(first_seg.start, first_seg.start + path_len) % self._capacity
                    )
                    self._valid_starts[indices] = self._compute_valid_starts(path_len)

                for i in range(first_seg.start, first_seg.stop):
                    self.idx_to_horizon[i] = first_seg.stop - 1 # TODO: this is not correct if a second_seg is present

                for i in range(second_seg.start, second_seg.stop):
                    self.idx_to_horizon[i] = second_seg.stop - 1

        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
            self._top = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
            self._top = first_seg.stop
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + path_len)

    def sample_transitions(self, batch_size: int):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        idx = np.random.choice(self._transitions_stored, batch_size)
        if self._pixel_dim is not None:
            ret_dict = {}
            keys = set(self._buffer.keys())
            for key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                keys.remove(pixel_key)
                keys.remove(state_key)
                if self._buffer[state_key].shape[1] != 0:
                    ret_dict[key] = np.concatenate([self._buffer[pixel_key][idx], self._buffer[state_key][idx]], axis=1)
                else:
                    ret_dict[key] = self._buffer[pixel_key][idx]
            for key in keys:
                ret_dict[key] = self._buffer[key][idx]
            return ret_dict
        else:
            # TODO: we currently don't consider the case where there is two segments
            # TODO: we currently ignore the next obs, which means we might miss s_T
            if self.sample_goals:
                horizons = [self.idx_to_horizon[i] for i in idx]

                future_goals = []
                future_goals_prev = []
                # future_coords = []
                for start, horizon in zip(idx, horizons):
                    pos = np.arange(horizon - start)

                    if horizon == start:
                        future_goals.append(self._buffer['next_obs'][horizon])
                        future_goals_prev.append(self._buffer['next_obs'][horizon - 1])
                        # future_coords.append(self._buffer['next_coordinates'][horizon])
                        continue

                    if self.put_remaining_mass_on_last_state:
                        probs = self.discount ** pos
                        probs *= (1 - self.discount)
                        partial_mass = probs[:-1].sum()
                        probs[-1] = 1 - partial_mass
                        assert np.isclose(probs.sum(), 1), 'probs should sum to 1'
                    else:
                        probs = self.discount ** pos
                        probs /= probs.sum()
                    f_idx = np.random.choice(pos, size=1, p=probs).item() + 1
                    assert start + f_idx <= horizon, 'sampled idx should not exceed horizon!'
                    future_goals.append(self._buffer['obs'][start + f_idx])
                    future_goals_prev.append(self._buffer['obs'][start + f_idx - 1])
                    # future_coords.append(self._buffer['coordinates'][start + f_idx])
                future_goals = np.stack(future_goals, axis=0)
                future_goals_prev = np.stack(future_goals_prev, axis=0)
                # future_coords = np.stack(future_coords, axis=0)

                results = {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}
                results['future_obs'] = future_goals
                results['future_obs_prev'] = future_goals_prev
                # results['future_coordinates'] = future_coords

                return results

            return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def sample_episodes(self, batch_size: int):
        """Sample a batch of **episodes** from the buffer.

        Args:
            batch_size (int): Number of episodes to sample.

        Returns:
            dict: A dict of arrays of shape (sampled_seq_len, batch_size, flat_dim).

        """
        sampled_episode_starts = self._sample_indices(batch_size)  # (B,)

        # get sequential indices
        indices = []
        for start in sampled_episode_starts:  # small loop
            end = start + self._sampled_seq_len  # continuous + T
            indices += list(np.arange(start, end) % self._capacity)

        # extract data
        batch = {key: buf_arr[indices] for key, buf_arr in self._buffer.items()}
        # each item has 2D shape (num_episodes * sampled_seq_len, dim)

        # generate masks (B, T)
        masks = self._generate_masks(indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = (
                batch[k]
                .reshape(batch_size, self._sampled_seq_len, -1)
                .transpose(1, 0, 2)
            )

        return batch
    
    def _generate_masks(self, indices, batch_size: int):
        """
        input: sampled_indices list of len B*T
        output: masks (B, T)
        """

        # get valid_starts of sampled sequences (B, T)
        # each row starts with a postive number, like 11111000, or 10000011, or 1s
        sampled_seq_valids = np.copy(self._valid_starts[indices]).reshape(
            batch_size, self._sampled_seq_len
        )
        sampled_seq_valids[sampled_seq_valids > 0.0] = 1.0  # binarize

        # build masks
        masks = np.ones_like(sampled_seq_valids, dtype=float)  # (B, T), default is 1

        # we want to find the boundary (ending) of sampled sequences
        # 	i.e. the FIRST 1 (positive number) after 0
        # 	this is important for varying length episodes
        # the boundary (ending) appears at the FIRST -1 in diff
        diff = sampled_seq_valids[:, :-1] - sampled_seq_valids[:, 1:]  # (B, T-1)
        # add 1s into the first column
        diff = np.concatenate([np.ones((batch_size, 1)), diff], axis=1)  # (B, T)

        # special case: the sampled sequence cannot cross self._top
        indices_array = np.array(indices).reshape(
            batch_size, self._sampled_seq_len
        )  # (B,T)
        # set the top as -1.0 as invalid starts
        diff[indices_array == self._top] = -1.0

        # now the start of next episode appears at the FIRST -1 in diff
        invalid_starts_b, invalid_starts_t = np.where(
            diff == -1.0
        )  # (1D array in batch dim, 1D array in seq dim)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                # for same batch_idx, we only care the first appearance of -1
                continue
            last_batch_index = batch_index

            invalid_indices = list(
                np.arange(start_index, self._sampled_seq_len)
            )  # to the end
            # extend to the list
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        # set invalids in the masks
        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks

    def _sample_indices(self, batch_size):
        # self._top points at the start of a new sequence
        # self._top - 1 is the end of the recently stored sequence
        valid_starts_indices = np.where(self._valid_starts > 0.0)[0]

        sample_weights = np.copy(self._valid_starts[valid_starts_indices])
        # normalize to probability distribution
        sample_weights /= sample_weights.sum()

        return np.random.choice(valid_starts_indices, size=batch_size, p=sample_weights)

    def _compute_valid_starts(self, seq_len: int):
        valid_starts = np.ones((seq_len), dtype=float)

        num_valid_starts = float(max(1.0, seq_len - self._sampled_seq_len + 1.0))

        # set the num_valid_starts: indices are zeros
        valid_starts[int(num_valid_starts) :] = 0.0

        return valid_starts

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity, array.shape[1]), array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self._path_segments.clear()
        self._buffer.clear()

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop > second.start

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)
