# -*- coding: utf-8 -*-
# Produkt = (Sekwencja, Protony, Elektrony) = (Seq, p, q)
# SINK jest jedynym produktem w którym Seq = ''.
# SINK oznacza dowolną cząsteczkę w której p = q.
SINK = ('', 0, 0, 'sink', 'sink')
CHARGE_RADIUS = 5
N_TERMINUS = '*'
# AA's include the ambiguity code:
AA = {'A', 'R', 'N', 'D', 'B', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'}
BASIC_AA = {"R", "K", "H", N_TERMINUS}
P = ('*ARNWK', 3, 0, 'precursor')
# CHARGE_FACTOR: function returning the charge-induced factor in the intensity of reaction
CHARGE_FACTOR = lambda(x): x**2


class molecule(object):
    """
    A class representing a molecule.

    Attributes
    -----
    sequence: str
        Aminoacid sequence, possibly with post-translational modifications.
        Additional NH2 groups denoted as '*'.
    charged: int
    quenched: int
        Charged and quenched protons.
    label: str
        Fragment name, one of 'precursor', 'cn' or 'zn', where 'n' is the number of aminoacids in the sequence.
    sequence_len: int
        Total number of aminoacids in the sequence (omitting e.g. NH2 groups).
    nb_of_basic_aa: int
        Number of basic aminoacids in the sequence.
    nb_of_breakable_bonds: int
        Number of breakable bonds in the sequence.
    Methods
    -----
    is_breakable(int): bool
    is_basic(int): bool

    """
    global BASIC_AA
    global INTSY

    def __init__(self, sequence, charged, quenched, ion_type):
        """
        :param sequence: str
        :param charged: int
        :param quenched: int
        :param ion_type: str
            One of 'c', 'cn', 'z', 'zn' or 'precursor'. Appropriate 'n' index, if absent, will be added automatically
            for fragment ions, so that the label has a form cn or zn, where n is the number of aminoacids.
        """
        self.sequence = sequence.upper()
        self.sequence_len = sum(x in AA for x in self.sequence)
        self._ion_type = ion_type.lower()
        if len(self._ion_type) == 1:
            assert self._ion_type in {'c', 'z'}, "Improper ion type: %s" % self._ion_type
            self.label = self._ion_type + str(self.sequence_len)
        elif self._ion_type[0] == 'c':
            assert int(self._ion_type[1:]) == self.sequence_len, "Improper labelling of c-ion: " \
                                                                 "%i aminoacids labeled as %s" % (self.sequence_len,
                                                                                                  self._ion_type)
            self.label = self._ion_type
        elif self._ion_type[0] == 'z':
            assert int(self._ion_type[1:]) == self.sequence_len, "Improper labelling of z-ion: " \
                                                                 "%i aminoacids labeled as %s"%(self.sequence_len,
                                                                                                self._ion_type)
            self.label = self._ion_type
        else:
            assert self._ion_type == 'precursor', "Improper ion type: %s" % self._ion_type
            self.label = self._ion_type
        self.charged = charged
        assert self.charged >= 0, "Negative charge."
        self.quenched = quenched
        assert self.quenched >= 0, "Negative number of quenched protons."
        self.nb_of_basic_aa = sum(x in BASIC_AA for x in sequence)
        assert self.quenched + self.charged <= self.nb_of_basic_aa, "Overcharged molecule."
        self._breakable_bonds = set(i for i in range(len(sequence)-1) if sequence[i+1] != 'P')
        self.nb_of_breakable_bonds = len(self._breakable_bonds)

    def get_ancestors(self, precursor, intensity_dict, cleavage_probabilities):
        """
        Return the ancestors of the current molecule.
        :param precursor: molecule
            Untouched precursor molecule.
        :return: list of tuples
            Returns a list of tuples (molecule, intensity), where molecule is an ancestral (substrate) molecule
            and intensity is the intensity of reaction leading from substrate to the current molecule.
        """
        ancestors = []
        intensities = []
        if self.sequence != precursor.sequence:
            dual_bases = precursor.nb_of_basic_aa-self.nb_of_basic_aa
            pmin = self.charged+1  # minimal substrate charge
            pmax = min(precursor.charged-self.quenched,  # quenched on product had to be quenched on substrate
                       self.charged+dual_bases+1,  # no overcharging of paired product ion
                       precursor.nb_of_basic_aa)  # no overcharging of the full sequence
            qmin = self.quenched
            qmax = min(precursor.charged-self.charged-1,  # charged had to be charged, plus reacting proton
                       self.quenched+dual_bases)  # no overcharging of paired ion

            for p in range(pmin, pmax+1):
                for q in range(qmin, min(precursor[1]-p+1, qmax+1)):
                    substrate = molecule(precursor.sequence, p, q, "precursor")
                    etd_intsy = INTSY('ETD', self, substrate, precursor, intensity_dict, cleavage_probabilities)
                    ancestors.append(substrate)
                    intensities.append(etd_intsy)

        # PTR occurs if the number of protons on the product is smaller than the one
        # on the precursor and there is place for another charge on the substrate.
        # If the substrate is a fragment, then there has to be additional charge availiable for ETD to occur
        # higher on the graph (a fragment can't take all the charges).
        if (self.label == 'precursor' and self.charged+self.quenched < min(precursor.charged, self.nb_of_basic_aa)) or \
                (self.label != 'precursor' and self.charged+self.quenched < min(precursor.charged-1, self.nb_of_basic_aa)):
            substrate = molecule(self.sequence, self.charged+1, self.quenched, self.label)
            ptr_intsy = INTSY('PTR', self, substrate, precursor, intensity_dict)
            ancestors.append(substrate)
            intensities.append(ptr_intsy)

        # ETnoD on a molecule could always happen if it has quenched protons, provided we won't get a fragment
        # that has charge of the precursor, because then ETD could not occur
        if product[2] > 0:
            substrate = (product[0], product[1]+1, product[2]-1, product[3])
            etnod_intsy = INTSY('ETnoD', product, substrate, precursor, intensity_dict)
            ancestors.append(substrate)
            intensities.append(etnod_intsy)


class precursor(molecule):
    """
    Class representing "untouched" precursor molecule, defining the experimental context.
    """


    def get_ancestors(self, precursor, intsy_dict, cleavage_probabilities):
        raise NotImplementedError("This is a helper class, not to be used like this. "
                                  "Use an appropriate method from the molecule class.")