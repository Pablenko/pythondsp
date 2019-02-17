import argparse

def bytes_sum(value, mask, count):
    masked = value & mask
    bytes_sum = 0

    for _ in range(0, count):
        val = masked & 1
        bytes_sum += val
        masked = masked >> 1

    return bytes_sum

def encoder(constraint, bits):
    state = 0
    output = ''
    all_mask = 2**constraint - 1
    first_last_mask = 1 | (1 << (constraint - 1))

    for b in bits:
        if b not in ['0', '1']:
            raise Exception("Wrong input format!")
        state = (int(b) << (constraint - 1)) | (state >> 1)
        all_bytes_sum = bytes_sum(state, all_mask, constraint)
        first_last_sum = bytes_sum(state, first_last_mask, constraint)

        output += '0' if all_bytes_sum % 2 == 0 else '1'
        output += '0' if first_last_sum % 2 == 0 else '1'

    return output


def decoder():
    pass

def main():
    parser = argparse.ArgumentParser(description='Viterbi encoder/decoder')
    parser.add_argument('--encode', action="store_true", help='do convolutional encoding')
    parser.add_argument('--decode', action="store_true", help='do Viterbi decoding')
    parser.add_argument('--c', nargs=1, type=int, help='constraint')
    parser.add_argument('--input', nargs=1, type=str, help='input bits')

    args = parser.parse_args()

    if args.encode:
        result = encoder(args.c[0], args.input[0])
        print result
    elif args.decode:
        decoder()
    else:
        print "Bye bye!"

if __name__ == "__main__":
    main()