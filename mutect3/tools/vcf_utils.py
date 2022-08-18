class VCFRecord:
    # initialize from one non-header line of an uncompressed VCF
    def __init__(self, vcf_line: str):
        self.tokens = vcf_line.split()

        self.filters = {} if (self.tokens[6] == '.' or self.tokens[6] == 'PASS') else set(self.tokens[6].split(';'))

    def contig(self):
        return self.tokens[0]

    def position(self):
        return int(self.tokens[1])

    def ref(self):
        return self.tokens[3]

    def alt(self):
        return self.tokens[4]

    def clear_filters(self):
        self.filters = {}
        self.tokens[6] = '.'

    def set_passing(self):
        self.filters = {'PASS'}
        self.tokens[6] = 'PASS'

    def add_filter(self, filter_name: str):
        self.filters.add(filter_name)
        self.tokens[6] = ';'.join(sorted(self.filters))





    # this presumes that self.tokens has been kept updated whenever a field is modified
    def as_string(self) -> str:
        return '\t'.join(self.tokens)
