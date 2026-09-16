def byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = visible[:]
    remaining = [b for b in range(256) if b not in visible]
    return {chr(c): b for b, c in zip(visible + remaining,
                                    codepoints + list(range(256, 256 + len(remaining))))}
