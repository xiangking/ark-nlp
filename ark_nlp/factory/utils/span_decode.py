def get_entity_bio(seq, id2label):
    """
    从BIO标注序列中解码出实体
    
    Args:
        seq (list): BIO标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            # output: [['PER', 0, 1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks


def get_entity_bios(seq, id2label):
    """
    从BIOS标注序列中解码出实体
    
    Args:
        seq (list): BIOS标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            get_entity_bios(seq)
            # output: [['PER', 0,1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]

    return chunks


def get_entities(seq, id2label=None, markup='bios'):
    """
    从标注序列中解码出实体
    
    Args:
        seq (list): 标注的序列
        id2label (dict): 标注ID对应的标签映射字典, 例如: {0: 'B-LOC', 1: 'I-LOC'}
        markup (dict): 序列标注方式: ['bio', 'bios']       
        
    Returns:
        list: (entity_type, entity_start_idx, entity_end_idx)三元组数组
        
    Example:
        .. code-block::
        
            seq = seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
            get_entities(seq)
            # output: [['PER', 0,1], ['LOC', 3, 3]]
    """  # noqa: ignore flake8"

    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    elif markup == 'bios':
        return get_entity_bios(seq, id2label)
    else:
        raise ValueError("The markup does not exist")
