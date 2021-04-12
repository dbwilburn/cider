## CIDer v1.0
## Usage: python CIDer.py (--nce nce) input.dlib (--output output)
## Returns input_CIDer.dlib

import numpy as np, pandas as pd
import sqlite3
import argparse
import os, pickle, sys, zlib, struct, re, itertools
from scipy import interpolate

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', 
                        type=str, 
                        help='path to input dlib')
    parser.add_argument('--output',
                        type=str,
                        help='path to output dlib',
                        default='CIDer_outpt.dlib')
    parser.add_argument('--nce',
                        type=int,
                        help='NCE value',
                        default=30)
    return parser.parse_args(args)
    
    
    

## Bring in model components
charge_pairs = [(2,1), (2,2), (3,1), (3,2)]
weights_dir = str(__file__).replace('CIDer.py','weights')
config = parse_args(sys.argv[1:])

weights_by_nce = pickle.load(open(os.path.join(weights_dir,'nce_weights.pkl'),'rb'))
weights = weights_by_nce[config.nce]

for ion_type in ['y','b']:
    weights[ion_type+'_mz_spline_curve'] = dict(zip( charge_pairs,
                                                     [interpolate.interp1d(weights[ion_type+'_mz_spline'][0], 
                                                                           weights[ion_type+'_mz_spline'][i]) 
                                                      for i in range(1,5)] ))

## Peptide classification functions
def mass_calc( seq, initial_mass = 18.0105647, mods={}, mod_offset=0 ):

    masses = { 'A':71.037113805,  'C':103.009184505, 'D':115.026943065, 'E':129.042593135,
               'F':147.068413945, 'G':57.021463735,  'H':137.058911875, 'I':113.084064015,
               'K':128.094963050, 'L':113.084064015, 'M':131.040484645, 'N':114.042927470,
               'P':97.052763875,  'Q':128.058577540, 'R':156.101111050, 'S':87.032028435,
               'T':101.047678505, 'V':99.068413945,  'W':186.079312980, 'Y':163.063328575 }

    mass = initial_mass
    for i in range(len(seq)):
        mass += masses[seq[i]]
        if i+1+mod_offset in mods:
            mass += mods[i+1+mod_offset]
    return mass

def mod_str_formatter(seq, mod_dict, start_site, end_site, mod_offset=0):
    output_str = ''
    for site in mod_dict:
        if start_site <= site <= end_site: # ensure that mods are not outside substring
            mod_str = str(int(site-mod_offset)) + seq[site-1-mod_offset] +\
                      format(int(mod_dict[site]), '+d') + ' '
            output_str += mod_str
    return output_str[:-1]

def mod_seq_generator(seq, mod_dict):
    mod_seq = ''
    start = 0
    for i in mod_dict:
        mod = '['+str(mod_dict[i])+']'
        mod_seq = mod_seq + seq[start:i] + mod
        start = i
    mod_seq += seq[start:]
    return mod_seq

def modseq_toModDict(mod_seq):
    mods = {}
    temp_seq = str(mod_seq)
    for i in range(mod_seq.count('[')):
        mod_start_index = temp_seq.find('[')
        mod_end_index = temp_seq.find(']')
        mod_mass = np.float64( temp_seq[mod_start_index+1:mod_end_index] )
        mods[mod_start_index] = mod_mass
        temp_seq = temp_seq[:mod_start_index] + temp_seq[mod_end_index+1:]
    return mods

def fragment_ion_generator(mod_seq, charge=2, main_only=False):
    p_mass = 1.00727646688
    seq = re.sub(r'\[.+?\]','',mod_seq)

    ## Identify modifications
    mods = modseq_toModDict(mod_seq)

    seq_len = len(seq)
    fragments = []
    # Ions will be tuples of (name, m/z, charge, sequence, modifications, type)
    # Start with b and y, then a, then internal -- only add if masses don't already exist
    for z in range(1, charge+1):
        if z ==1: 
            charge_state = 'single'
        elif z == 2:  
            charge_state = 'double'
        else:  
            charge_state = 'multiple'
        for i in range(1, seq_len):
            # b ion
            ab_seq = seq[:i]
            b_mass = mass_calc( ab_seq, initial_mass = 0, mods = mods )
            fragments.append( ( 'b'+str(i)+'_'+str(z)+'+', #name
                                (b_mass+z*p_mass)/z,
                                z, 
                                ab_seq, 
                                mod_str_formatter(ab_seq, mods, 1, i),
                                'expected_'+charge_state ) )
            # y ion
            if i < seq_len:
                y_seq = seq[i:]
                y_mass = mass_calc( y_seq, initial_mass = 18.0105647,
                                    mods = mods, mod_offset = i )
                fragments.append( ( 'y'+str(seq_len-i)+'_'+str(z)+'+', #name
                                   (y_mass+z*p_mass)/z,
                                   z,
                                   y_seq, 
                                   mod_str_formatter(y_seq, mods, i+1, seq_len, i),
                                   'expected_'+charge_state ) )
        
        if not main_only:
            for i in range(1, seq_len):
                # a ion
                ab_seq = seq[:i]
                a_mass = mass_calc( ab_seq, initial_mass = -27.99491463, mods = mods )
                a_mz = (a_mass+z*p_mass)/z
                if not np.any( np.isclose(a_mz, [x[1] for x in fragments]) ):
                    fragments.append( ( 'a'+str(i)+'_'+str(z)+'+', #name
                                        a_mz,
                                        z, 
                                        ab_seq, 
                                        mod_str_formatter(ab_seq, mods, 1, i),
                                        'modified_'+charge_state ) )
            for i in range(1, seq_len):
                for j in range(i+2, seq_len):
                    i_seq = seq[i:j]
                    i_mass = mass_calc( i_seq, initial_mass = 0, 
                                        mods = mods, mod_offset = i )
                    i_mz = (i_mass+z*p_mass)/z
                    if not np.any( np.isclose(i_mz, [x[1] for x in fragments]) ):
                        fragments.append( ( 'i'+str(i+1)+'-'+str(j)+'_'+str(z)+'+', #name
                                            i_mz,
                                            z,
                                            i_seq,
                                            mod_str_formatter(i_seq, mods, i+1, j, i),
                                            'internal_'+charge_state ) )
    
    df = pd.DataFrame( fragments, columns=['ion', 'predict_mz', 'z', 'sequence', 'mods', 'category'] )
    df = df.sort_values('predict_mz')
    return df

def y_fitter(row, labels, p_z, y_z):
    values = dict(zip(labels,row))
    y_fit = weights['y_mz_spline_curve'][(p_z,y_z)]( (values['y_mz']+(y_z-1)*1.00727646688)/y_z ) +\
            weights['y_pn'][p_z][y_z][values['peplength']] +\
            weights['y_yRcount_by_yn'][p_z][y_z][values['y_R_count']][values['y_number']] +\
            weights['y_yKcount_by_yn'][p_z][y_z][values['y_K_count']][values['y_number']] +\
            weights['y_yHcount_by_yn'][p_z][y_z][values['y_H_count']][values['y_number']] +\
            weights['y_bRcount_by_yn'][p_z][y_z][values['b_R_count']][values['b_number']] +\
            weights['y_bKcount_by_yn'][p_z][y_z][values['b_K_count']][values['b_number']] +\
            weights['y_bHcount_by_yn'][p_z][y_z][values['b_H_count']][values['b_number']] +\
            weights['y_side_weights']['left'][p_z][y_z][values['left_residue']] +\
            weights['y_side_weights']['right'][p_z][y_z][values['right_residue']]
    return np.power(2, float(y_fit))

def b_fitter(row, labels, p_z, y_z, b_z):
    values = dict(zip(labels,row))
    b_fit = weights['b_mz_spline_curve'][(p_z,y_z)]((values['b_mz']+(b_z-1)*1.00727646688)/b_z) +\
            weights['b_pn'][p_z][y_z][values['peplength']] +\
            weights['b_bRcount_by_bn'][p_z][y_z][values['b_R_count']][values['b_number']] +\
            weights['b_bKcount_by_bn'][p_z][y_z][values['b_K_count']][values['b_number']] +\
            weights['b_bHcount_by_bn'][p_z][y_z][values['b_H_count']][values['b_number']] +\
            weights['b_yRcount_by_bn'][p_z][y_z][values['y_R_count']][values['b_number']] +\
            weights['b_yKcount_by_bn'][p_z][y_z][values['y_K_count']][values['b_number']] +\
            weights['b_yHcount_by_bn'][p_z][y_z][values['y_H_count']][values['b_number']] +\
            weights['b_side_weights']['left'][p_z][y_z][values['left_residue']] +\
            weights['b_side_weights']['right'][p_z][y_z][values['right_residue']]
    return np.power(2,float(b_fit))


def cider_predict(row, labels):
    peptide = dict(zip(labels,row))
    mzs = np.ndarray(shape=(int(peptide['MassEncodedLength']/8)), dtype='>d', 
                     buffer=zlib.decompress(peptide['MassArray'])).astype('float64')
    ints = np.ndarray(shape=(int(peptide['IntensityEncodedLength']/4)), dtype='>f', 
                      buffer=zlib.decompress(peptide['IntensityArray'])).astype('float64')
    norm_ints = ints / np.max(ints)
    
    predict = fragment_ion_generator(peptide['PeptideModSeq'], 
                                     charge=peptide['PrecursorCharge'],
                                     main_only=True)
    
    fragment_ints = []
    for ion_mz in predict.predict_mz:
        # Find the index of the closest m/z in mzs
        idx = np.abs(mzs-ion_mz).argmin()
        # Check if within tolerance
        tol = ion_mz*2e-5 # 20 ppm
        if np.abs(mzs[idx] - ion_mz) <= tol and ints[idx] > 0.01:
            fragment_ints.append( norm_ints[idx] )
        else:
            fragment_ints.append( 0.01 )
    predict['intensity'] = fragment_ints

    # Pivot the table so that b and y ions are paired
    predict['ion_number'] = [int(x[1:].split('_')[0]) for x in predict.ion]
    predict['ion_number_rev'] = len(peptide['PeptideSeq']) - predict.ion_number
    predict['ion_charge'] = [int(x[:-1].split('_')[1]) for x in predict.ion]
    predict['ion_type'] = [x[0] for x in predict.ion]
   
    if peptide['PrecursorCharge'] not in [2,3]:
        print('Cannot calculate prediction for '+peptide['PeptideModSeq']+'_'+
              str(peptide['PrecursorCharge'])+'+')
        return 0
    else:
        b1_df = predict.copy()[(predict.ion_type == 'b') & (predict.ion_charge == 1)]
        b1_df = b1_df.rename(columns={'ion':'b_ion', 'predict_mz':'b_mz', 'z':'b_z', 'ion_number':'b_number',
                                      'sequence':'b_sequence', 'intensity':'b_1+_intensity'})
        b2_df = predict.copy()[(predict.ion_type == 'b') & (predict.ion_charge == 2)]
        b2_df = b2_df.rename(columns={'ion_number':'b_number', 'intensity':'b_2+_intensity'})
        b_df = pd.merge(b1_df, b2_df[['b_number','b_2+_intensity']], on='b_number')
        
        
        y1_df = predict.copy()[(predict.ion_type == 'y') & (predict.ion_charge == 1)]
        y1_df = y1_df.rename(columns={'ion':'y_ion', 'predict_mz':'y_mz', 'z':'y_z', 'ion_number':'y_number',
                                      'sequence':'y_sequence', 'intensity':'y_1+_intensity'})
        y2_df = predict.copy()[(predict.ion_type == 'y') & (predict.ion_charge == 2)]
        y2_df = y2_df.rename(columns={'ion_number':'y_number', 'intensity':'y_2+_intensity'})
        y_df = pd.merge(y1_df, y2_df[['y_number','y_2+_intensity']], on='y_number')

        pep_df = pd.merge(b_df[['b_ion', 'b_mz', 'b_z', 'b_number', 'b_sequence', 
                                'b_1+_intensity', 'b_2+_intensity']], 
                          y_df[['y_ion', 'y_mz', 'y_z', 'y_number', 'y_sequence', 
                                'y_1+_intensity','y_2+_intensity', 'ion_number_rev']],
                          left_on='b_number', right_on='ion_number_rev')
        
        ## Scale the intensities according to the most abundant ion
        max_val = np.max( list(pep_df['y_1+_intensity']) + list(pep_df['y_2+_intensity']) +
                          list(pep_df['b_1+_intensity']) + list(pep_df['b_2+_intensity']) )
        for i,z in itertools.product(['y','b'],[1,2]):
            pep_df[i+'_'+str(z)+'+_intensity_scaled'] = pep_df[i+'_'+str(z)+'+_intensity']/max_val

            
        pep_df['peplength'] = len(peptide['PeptideSeq'])
        pep_df['left_residue'] = [x[-1] for x in pep_df.b_sequence]
        pep_df['right_residue'] = [x[0] for x in pep_df.y_sequence]
        for i in ['y','b']:
            for r in ['R','K','H']: pep_df[i+'_'+r+'_count'] = [x.count(r) for x in pep_df[i+'_sequence']]
        
        ## Soooo I guess I have the peptide df. Now to just apply the model
        model_fits = {'mz':[], 'int':[]}
        for y_z in [1,2]:
            if peptide['PrecursorCharge'] == 2: b_z = y_z
            else: b_z = [x for x in [1,2] if x != y_z][0]

            frag_columns = ['index'] + list(pep_df.columns)
            y_wgts = [y_fitter(x, frag_columns, peptide['PrecursorCharge'], y_z) for x in pep_df.itertuples()]
            y_ints = pep_df['y_'+str(y_z)+'+_intensity_scaled'] * y_wgts
            b_wgts = np.array([b_fitter(x, frag_columns, peptide['PrecursorCharge'], y_z, b_z) for x in pep_df.itertuples() ])
            if peptide['PrecursorCharge'] == 2 and b_z == 2:
                b_ints = 0.01 * b_wgts
            else: b_ints = y_ints * b_wgts

            y_mz = (pep_df.y_mz +(y_z-1)*1.00727646688)/y_z
            b_mz = (pep_df.b_mz +(b_z-1)*1.00727646688)/b_z
            model_fits['mz'] += list(y_mz) + list(b_mz)
            model_fits['int'] += list(y_ints) + list(b_ints)
        pred_mzs = bytearray()
        pred_ints = bytearray()
        for mz in model_fits['mz']: pred_mzs += bytearray(struct.pack('>d', mz))
        for i in model_fits['int']: pred_ints += bytearray(struct.pack('>f', i))
        peptide['MassArray'] = zlib.compress(pred_mzs)
        peptide['MassEncodedLength'] = len(model_fits['mz']) * 8
        peptide['IntensityArray'] = zlib.compress(pred_ints)
        peptide['IntensityEncodedLength'] = len(model_fits['int']) * 4

        return list(peptide.values())


## Read in dlib
input_dlib = config.input_file
con = sqlite3.connect(input_dlib)
df = pd.read_sql_query('SELECT * from entries', con)
con.close()

## Loop over each row, extract spectra, annotate, and apply model
new_fit = []
labels = list(df.columns)
for row in df.itertuples(index=False):
    new_row = cider_predict(row, labels)
    if new_row != 0:    new_fit.append(new_row)
new_df = pd.DataFrame(new_fit, columns=labels)

output_dlib = config.output
if os.path.isfile(output_dlib): os.remove(output_dlib)
con = sqlite3.connect(output_dlib)
cursor = con.cursor()
cursor.execute('''CREATE TABLE metadata( Key string not null, Value string not null )''')
cursor.execute("INSERT INTO metadata (Key, Value) VALUES ('version', '0.1.14')")
new_df.to_sql('entries', con, if_exists='replace', index=False)
con.commit()
con.close()
