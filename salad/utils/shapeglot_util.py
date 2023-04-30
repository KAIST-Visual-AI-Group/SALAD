# References: https://github.com/optas/shapeglot
#             https://github.com/63days/PartGlot.

from six.moves import cPickle


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: a generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """

    in_file = open(file_name, "rb")
    if python2_to_3:
        size = cPickle.load(in_file, encoding="latin1")
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding="latin1")
        else:
            yield cPickle.load(in_file)
    in_file.close()


def get_mask_of_game_data(
    game_data: DataFrame,
    word2int: Dict,
    only_correct: bool,
    only_easy_context: bool,
    max_seq_len: int,
    only_one_part_name: bool,
):
    """
    only_correct (if True): mask will be 1 in location iff human listener predicted correctly.
    only_easy (if True): uses only easy context examples (more dissimilar triplet chairs)
    max_seq_len: drops examples with len(utterance) > max_seq_len
    only_one_part_name (if True): uses only utterances describing only one part in the give set.
    """
    mask = np.array(game_data.correct)
    if not only_correct:
        mask = np.ones_like(mask, dtype=np.bool)

    if only_easy_context:
        context_mask = np.array(game_data.context_condition == "easy", dtype=np.bool)
        mask = np.logical_and(mask, context_mask)

    short_mask = np.array(
        game_data.text.apply(lambda x: len(x)) <= max_seq_len, dtype=np.bool
    )
    mask = np.logical_and(mask, short_mask)

    part_indicator, part_mask = get_part_indicator(game_data.text, word2int)
    if only_one_part_name:
        mask = np.logical_and(mask, part_mask)

    return mask, part_indicator
