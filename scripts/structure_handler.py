from scripts import *

desc_be_removed = ['SMR_VSA8', 'SlogP_VSA9', 'fr_prisulfonamd', 'MaxAbsPartialCharge', 'MinPartialCharge',
                   'MaxPartialCharge', 'MinAbsPartialCharge']


class ChemCurator:

    def __init__(self, smiles_col, df):
        """
        A DataFrame with a SMILES column is passed.
        :param smiles_col:
        :param df:
        """
        self.smiles_col = smiles_col
        self.df = df
        self.metal_index = []
        self.frag_index = []
        self.non_c_index = []

    def rem_non_c(self):
        """

        :return:
        """

        df_no_c = self.df[self.smiles_col][~self.df[self.smiles_col].str.contains('.*c|C[^.l].*')]
        self.non_c_index = df_no_c.index
        self.df = self.df.drop(self.non_c_index, axis=0)

    def rem_met(self):
        """

        :return:
        """
        # list of metals, list of salts, halogenes
        met_list = ['Be', '.*B[^.r].*', 'Al', 'Si', 'Ti', 'V', 'Cr',
                    'Co', 'Ni', 'Cu', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
                    'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Ba', 'Hf', 'Ta',
                    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
                    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Fl', 'Lv', 'La', 'Ce', 'Pr', 'Nd',
                    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U',
                    'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Zn', 'Mn', 'Fe', ]
        salt_metals = ['Na', 'Mg', 'K', 'Ca', 'Li']
        # extra = ['Cn', 'Cs', 'Sc']
        # hal_list = ['F', 'Cl', 'Br', 'I', 'S', 'P']

        metal_df = self.df[self.smiles_col][
            self.df[self.smiles_col].replace(np.nan, 'error').str.contains('|'.join(met_list + salt_metals),
                                                                           case=True, regex=True)]
        self.metal_index = metal_df.index
        self.df = self.df.drop(self.metal_index, axis=0)

    def rem_frags(self):
        """

        :return:
        """

        frags_col = self.df[self.smiles_col].apply(lambda x: self.frags(x))
        frags_count = self.df[self.smiles_col].apply(lambda x: self.frag_counter(x))
        frags_rest = self.df[self.smiles_col].apply(lambda x: self.frag_counter_rest(x))

        # add frags columns to df, amount of frags, list of frag length, amount of frags after removing the ones
        check_frags_df = self.df[[self.smiles_col]]
        check_frags_df['frags_col'] = frags_col
        check_frags_df['frags_count'] = frags_count
        check_frags_df['frags_rest'] = frags_rest
        self.frag_index = check_frags_df[check_frags_df.frags_rest > 2].index.tolist()
        self.df = self.df.drop(self.frag_index, axis=0)

    # functions for fragments

    @staticmethod
    def frags(smile):
        """

        :param smile:
        :return:
        """
        try:
            fr = GetMolFrags(Chem.MolFromSmiles(str(smile)))
            return len(fr)
        except:
            return None

    @staticmethod
    def frag_counter(x):
        """
        Get frags and return length of each frag
        :param x:
        :return:
        """

        try:

            fr = GetMolFrags(Chem.MolFromSmiles(str(x)))
            lens = [len(x) for x in fr]
            return lens
        except:
            return None

    @staticmethod
    def frag_counter_rest(x):
        """
        Get frags and return length of each frag if not 1
        :param x:
        :return:
        """

        try:
            fr = GetMolFrags(Chem.MolFromSmiles(str(x)))
            lens = [len(x) for x in fr if len(x) != 1]
            return len(lens)
        except:
            return None


def calc_desc_fp(df, option):
    """

    :param df: needs a clean .smiles columns
    :return:
    """
    # TODO could also go with MACCS
    # generate 2D desc from RDKit
    df_desc_2d = getDescDf(df.smiles.apply(lambda x: compDesc(x)))
    # df_desc_2d.drop(desc_be_removed, axis=1, inplace=True)

    # generate 3D desc from RDKit
    df_desc_3d = df.smiles.apply(lambda x: SmiTo3D(x))
    missing_3d_index = df_desc_3d[df_desc_3d.isnull().sum(axis=1) > 1].index
    df_w_dropped_fails_f_3d = df.drop(missing_3d_index, axis=0)

    # concat desc
    desc_all = pd.concat([df_desc_3d, df_desc_2d], axis=1).loc[df_w_dropped_fails_f_3d.index]
    desc_all.drop(desc_be_removed, axis=1, inplace=True)

    # gen Morgan FPs
    df_fp = df_w_dropped_fails_f_3d.smiles.apply(smfp)
    if option == 0:

        df_fp.columns = ['fp' + str(i) for i in df_fp.columns]
    elif option == 1:

        df_fp.columns = ['fp' + str(i + 1) for i in df_fp.columns]
    else:
        print('Option error')

    return df_fp, desc_all


def currate_smiles_columns(df: pd.DataFrame, smile_column_name: str):
    """
    Cleaning smiles in a dataframe from imported ChemCurator
    :param df:
    :param smile_column_name:
    :return:
    """
    df.rename(columns={smile_column_name: 'smiles'})
    dc = ChemCurator('smiles', df)
    dc.rem_non_c()
    dc.rem_met()
    dc.rem_frags()
    # drop invalids
    dfout = df.drop(dc.metal_index.tolist() + dc.non_c_index.tolist() + dc.frag_index, axis=0)
    # check smiles
    smiles_Test = dfout.smiles.apply(check_sms)
    # smiles which are drooped
    smi_drop_index = smiles_Test[smiles_Test.isnull()].index
    dfout.drop(smi_drop_index, axis=0, inplace=True)

    print(f'{df.shape[0] - dfout.shape[0]} were dropped in the cleaning process.')
    return dfout


def smi2fp(smiles, rad=3, bits=5120):
    """

    :param smiles: smiles to calc fingerprints from
    :param rad: radius
    :param bits: length of vector, 1024,2048,4096,5120
    :return: pandas series of the bit vector
    """

    # prep dict and array
    bi = {}
    bv = np.zeros(bits)
    # convert to mol and calc FP
    mol_for_fp = Chem.MolFromSmiles(smiles)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol_for_fp, radius=rad, bitInfo=bi, nBits=bits)
    # convert to array and series
    DataStructs.ConvertToNumpyArray(fp, bv)
    bs = pd.Series(bv)

    return bs


def _cansmi(smiles: str):
    '''
    function created because of typerror appearing
    :param smiles: smiles
    :return: canonical smiles if possible
    '''
    try:
        return Chem.CanonSmiles(smiles)
    except TypeError:
        return None


def compDesc(smiles):
    """

    :param smiles:
    :return:
    """
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    descriptors = list(np.array(Descriptors._descList)[:, 0])
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
    try:
        cansmi = _cansmi(smiles)
        mol = Chem.MolFromSmiles(cansmi)
        res = calculator.CalcDescriptors(mol)
    except:
        res = None
    return res


def getDescDf(arr):
    """

    :param arr:
    :return:
    """
    from rdkit.Chem import Descriptors
    descriptors = list(np.array(Descriptors._descList)[:, 0])
    X = pd.Series(arr).apply(lambda x: pd.Series(x))
    X.columns = descriptors
    return X


def Get3DMolFromSmiles(smiles):
    """

    :param smiles:
    :return:
    """
    m3 = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        m3 = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m3, useRandomCoords=False, randomSeed=42)
        m3 = Chem.RemoveHs(m3)
    except Exception as error:
        pass

    return m3


def SmiTo3D(smiles):
    """

    :param smiles:
    :return:
    """
    return pd.Series(get3ddescriptor(Get3DMolFromSmiles(smiles)))


def get3ddescriptor(rdkit_mol):
    """

    :param rdkit_mol:
    :return:
    """
    from rdkit.Chem import Descriptors3D
    from rdkit.Chem import rdMolDescriptors

    try:
        asphericity = Descriptors3D.Asphericity(rdkit_mol)  # single float
    except:
        asphericity = np.nan
    try:
        eccentricity = Descriptors3D.Eccentricity(rdkit_mol)  # single float
    except:
        eccentricity = np.nan
    try:
        inertialshapefactor = Descriptors3D.InertialShapeFactor(rdkit_mol)  # single float
    except:
        inertialshapefactor = np.nan
    try:
        npr1 = Descriptors3D.NPR1(rdkit_mol)  # single float
    except:
        npr1 = np.nan
    try:
        npr2 = Descriptors3D.NPR2(rdkit_mol)  # single float
    except:
        npr2 = np.nan
    try:
        pmi1 = Descriptors3D.PMI1(rdkit_mol)  # single float
    except:
        pmi1 = np.nan
    try:
        pmi2 = Descriptors3D.PMI2(rdkit_mol)  # single float
    except:
        pmi2 = np.nan
    try:
        pmi3 = Descriptors3D.PMI3(rdkit_mol)  # single float
    except:
        pmi3 = np.nan
    try:
        radiusofgyration = Descriptors3D.RadiusOfGyration(rdkit_mol)  # radius of gyration   # single float
    except:
        radiusofgyration = np.nan
    try:
        spherocityindex = Descriptors3D.SpherocityIndex(rdkit_mol)  # single float
    except:
        spherocityindex = np.nan
    try:
        pdf = rdMolDescriptors.CalcPBF(rdkit_mol)  # Returns the PBF (plane of best fit) descriptor   # single float
    except:
        pdf = np.nan

    value_list = [asphericity, eccentricity, inertialshapefactor, npr1, npr2, pmi1, pmi2, pmi3, radiusofgyration,
                  spherocityindex, pdf]
    descriptorNames = ["asphericity", "eccentricity", "inertialshapefactor", "npr1", "npr2", "pmi1", "pmi2", "pmi3",
                       "radiusofgyration", "spherocityindex", "calcpdf"]
    values = dict(zip(descriptorNames, value_list))

    return values


def smfp(x):
    """

    :param x:
    :return:
    """
    try:
        return smi2fp(x)
    except:
        return None


def check_sms(smi):
    """
    Check smiles circular
    :param smi:
    :return:
    """
    from rdkit import Chem
    try:
        x = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except:
        x = np.nan
    return x
