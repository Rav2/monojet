from sklearn import preprocessing 
import logging
import awkward0 as awkward 
import numpy as np
from keras.utils import to_categorical
import math
import copy


def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)


def pad_array(a, maxlen, value=0., dtype='float32'):
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    return x


class Dataset(object):

    def __init__(self, filepath, feature_dict = {}, label='label', pad_len=300, data_format='channel_first', simple_mode=False):
        self.filepath = filepath
       
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt_log', 'part_e_log', 'part_etarel', 'part_phirel','origin']
            feature_dict['mask'] = ['part_pt_log']
            feature_dict['origin'] = ['origin']
            feature_dict['track'] = ['track_pt', 'track_eta', 'track_phi']
            feature_dict['photon'] = ['photon_ET', 'photon_eta', 'photon_phi']
            feature_dict['hadron'] = ['hadron_ET', 'hadron_eta', 'hadron_phi']
            feature_dict['jet'] = ['jet_PT', 'jet_Eta', 'jet_Phi', 'jet_Mass', 'jet_Flavor']
            feature_dict['squark'] = ['squark_pt']
            feature_dict['jet1_con'] = ['con_jet1_phi_arr', 'con_jet1_eta_arr', 'con_jet1_pt_arr', 'con_jet1_type_arr']
            feature_dict['jet2_con'] = ['con_jet2_phi_arr', 'con_jet2_eta_arr', 'con_jet2_pt_arr', 'con_jet2_type_arr']
            feature_dict['jet3_con'] = ['con_jet3_phi_arr', 'con_jet3_eta_arr', 'con_jet3_pt_arr', 'con_jet3_type_arr']
        self.feature_dict = copy.copy(feature_dict)
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._label = None
        self._ev_phi = None
        self._ev_phi_raw = None
        self._ev_eta = None
        self_event_eta_raw = None
        self._ev_met = None
        self._ev_met_raw = None
        self._ev_MT2 = None
        self._ev_MT2_raw = None
        self._ev_ht = None
        self._ev_m = None
        self._jet_flavor = None
        self._one_hot = None
        self._Dphi = None
        self._simple_mode = simple_mode
        # load data
        self._load()
        
        new_arr = []
        for ev in self._values['origin']:
            ev_arr = to_categorical(ev, num_classes=3) 
            new_arr.append(ev_arr)
        self._one_hot = new_arr

        # phi diff for jets
        jet_phi = self._values['jet'][:, :, 2]
        ev_phi = self._ev_phi_raw
        ev_phi = np.repeat(ev_phi[:, np.newaxis], 4, axis=1)
        Dphi = jet_phi - ev_phi
        cond_up = Dphi >= math.pi
        Dphi[cond_up] =  Dphi[cond_up] - 2*math.pi
        cond_down = Dphi < -1*math.pi
        Dphi[cond_down] =  Dphi[cond_down] + 2*math.pi
        # zero Dphi for jets that are not there
        Dphi[ self._values['jet'][:, :, 0] < 1.0 ] = 0.0
        self._Dphi = Dphi
        
        # calculate log of jet pT
        #jet_log_pt = np.expand_dims( np.log(self._values['jet'][:, :, 0]).clip(-11, 11), -1)
        #new_jet_dat = np.concatenate( (self._values['jet'], jet_log_pt), axis=2)
        #self._values['jet'] = new_jet_dat
        #self.feature_dict['jet'].append('jet_PT_log')
        print('Dataset created!')
        
    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            print('Keys in the datafile found:')
            print(list(a.keys()))
            self._label = a[self.label]
            self._ev_phi = a['event_phi']
            self._ev_phi_raw = a['event_phi']
            self._ev_eta = a['event_eta']
            self._ev_eta_raw = a['event_eta']
            self._ev_met = a['event_MET']
            self._ev_met_raw = a['event_MET']
            self._ev_MT2 = a['event_MT2']
            self._ev_MT2_raw = a['event_MT2']
            self._ev_ht = a['event_HT']
            self._ev_m = a['event_mass']

            if self._simple_mode == True:
                for k in self.feature_dict:
                    logging.info('Loading {}'.format(k))
#                     if k in ['track', 'photon', 'hadron', 'jet', 'squark', 'jet1_con', 'jet2_con', 'jet3_con']:
#                         continue
                    cols = self.feature_dict[k]
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    arrs = []
                    for col in cols:
                        logging.info('Loading %s' % cols)
                        if counts is None:
                            counts = a[col].counts
                        else:
                            assert np.array_equal(counts, a[col].counts)
                        arrs.append(pad_array(a[col], self.pad_len))
                    self._values[k] = np.stack(arrs, axis=self.stack_axis)
            else:
                for k in self.feature_dict:
                    logging.info('Loading {}'.format(k))
                    if k in ['track', 'photon', 'hadron', 'jet', 'squark', 'jet1_con', 'jet2_con', 'jet3_con']:
                        counts = None
                    cols = self.feature_dict[k]
                    if not isinstance(cols, (list, tuple)):
                        cols = [cols]
                    arrs = []
                    for col in cols:
                        logging.info('Loading %s' % cols)
                        print('Loading %s' % cols)
                        if k != 'squark':
                            if counts is None:
                                counts = a[col].counts
                            else:
                                #print(counts[:3], a[col].counts[:3])
                                assert np.array_equal(counts, a[col].counts)
                        if k == 'jet':
                            arrs.append(pad_array(a[col], 4))
                        elif k == 'squark':
                            try:
                                arrs.append(pad_array(a[col], 3))
                            except KeyError:
                                arrs.append(pad_array([], 3))
                        elif k in ('jet1_con', 'jet2_con', 'jet3_con'):
                            arrs.append(pad_array(a[col], 80))
                        else:
                            arrs.append(pad_array(a[col], self.pad_len))


                    self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        elif key=='event_phi':
            return self._ev_phi
        elif key=='event_phi_raw':
            return self._ev_phi_raw
        elif key=='event_eta':
            return self._ev_eta
        elif key=='event_eta_raw':
            return self._ev_eta_raw
        elif key=='event_met':
            return self._ev_met
        elif key=='event_met_raw':
            return self._ev_met_raw
        elif key=='event_MT2':
            return self._ev_MT2
        elif key=='event_MT2_raw':
            return self._ev_MT2_raw
        elif key=='event_ht':
            return self._ev_ht
        elif key=='event_m':
            return self._ev_m
        elif key=='Dphi':
            return self._Dphi
        else:
            return self._values[key]

    def normalize_eventlevel_data(self, scalers = None):
        scalers_dict = {}
        # event phi
        logging.info('Normalizing ev_phi')
        if scalers is None:
            scaler_phi = preprocessing.StandardScaler().fit(self._ev_phi.reshape(-1, 1))
            scalers_dict['scaler_phi'] = scaler_phi
        else:
            scaler_phi = scalers['scaler_phi']
        self._ev_phi = scaler_phi.transform(self._ev_phi.reshape(-1, 1)).flatten()
        # event eta
        logging.info('Normalizing ev_eta')
        if scalers is None:
            scaler_eta = preprocessing.StandardScaler().fit(self._ev_eta.reshape(-1, 1))
            scalers_dict['scaler_eta'] = scaler_eta
        else:
            scaler_eta = scalers['scaler_eta']
        self._ev_eta = scaler_eta.transform(self._ev_eta.reshape(-1, 1)).flatten()
        # event MET
        logging.info('Normalizing ev_met')
        if scalers is None:
            scaler_met = preprocessing.StandardScaler().fit(self._ev_met.reshape(-1, 1))
            scalers_dict['scaler_met'] = scaler_met
        else:
            scaler_met = scalers['scaler_met']
        self._ev_met = scaler_met.transform(self._ev_met.reshape(-1, 1)).flatten()
         # event MT2
        logging.info('Normalizing ev_MT2')
        if scalers is None:
            scaler_MT2 = preprocessing.StandardScaler().fit(self._ev_MT2.reshape(-1, 1))
            scalers_dict['scaler_MT2'] = scaler_MT2
        else:
            scaler_MT2 = scalers['scaler_MT2']
        self._ev_MT2 = scaler_MT2.transform(self._ev_MT2.reshape(-1, 1)).flatten()
        # event HT
        logging.info('Normalizing ev_ht')
        if scalers is None:
            scaler_ht = preprocessing.StandardScaler().fit(self._ev_ht.reshape(-1, 1))
            scalers_dict['scaler_ht'] = scaler_ht
        else:
            scaler_ht = scalers['scaler_ht']
        self._ev_ht = scaler_ht.transform(self._ev_ht.reshape(-1, 1)).flatten()
        # event mass
        logging.info('Normalizing ev_m')
        if scalers is None:
            scaler_m = preprocessing.StandardScaler().fit(self._ev_m.reshape(-1, 1))
            scalers_dict['scaler_m'] = scaler_m
        else:
            scaler_m = scalers['scaler_m']
        self._ev_m = scaler_m.transform(self._ev_m.reshape(-1, 1)).flatten()

        if scalers is None:
            return scalers_dict
        else:
            return scalers
    
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label
    
    @property
    def part_origin(self):
         return self._jet_flavor

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]
        self._ev_phi = self._ev_phi[shuffle_indices]
        self._ev_eta = self._ev_eta[shuffle_indices]
        self._ev_phi_raw = self._ev_phi_raw[shuffle_indices]
        self._ev_eta_raw = self._ev_eta_raw[shuffle_indices]
        self._ev_met = self._ev_met[shuffle_indices]
        self._ev_met_raw = self._ev_met_raw[shuffle_indices]
        self._ev_MT2 = self._ev_MT2[shuffle_indices]
        self._ev_MT2_raw = self._ev_MT2_raw[shuffle_indices]
        self._ev_ht = self._ev_ht[shuffle_indices]
        self._ev_m = self._ev_m[shuffle_indices]
        
    def normalize_jetlevel_data(self, scalers = None):
        scalers_dict = {}
  
        for ii in range(0, 4):
            logging.info(f'Normalizing jet{ii+1}')
            if scalers is None:
                scaler_pt = preprocessing.StandardScaler().fit(self.X['jet'][:,ii,0].reshape(-1, 1))
                scalers_dict[f'scaler_jet{ii+1}PT'] = scaler_pt
                scaler_eta = preprocessing.StandardScaler().fit(self.X['jet'][:,ii,1].reshape(-1, 1))
                scalers_dict[f'scaler_jet{ii+1}ETA'] = scaler_eta
                scaler_mass = preprocessing.StandardScaler().fit(self.X['jet'][:,ii,3].reshape(-1, 1))
                scalers_dict[f'scaler_jet{ii+1}MASS'] = scaler_mass
                #scaler_pt_log = preprocessing.StandardScaler().fit(self.X['jet'][:,ii,5].reshape(-1, 1))
                #scalers_dict[f'scaler_jet{ii+1}PT_log'] = scaler_pt_log
                scaler_dphi = preprocessing.StandardScaler().fit(self._Dphi[:,ii].reshape(-1, 1))
                scalers_dict[f'scaler_jet{ii+1}DPHI'] = scaler_dphi
            else:
                scaler_pt = scalers[f'scaler_jet{ii+1}PT']
                scaler_eta = scalers[f'scaler_jet{ii+1}ETA']
                scaler_mass = scalers[f'scaler_jet{ii+1}MASS']
                #scaler_pt_log = scalers[f'scaler_jet{ii+1}PT_log']
                scaler_dphi = scalers[f'scaler_jet{ii+1}DPHI']
                
            self.X['jet'][:,ii,0] = scaler_pt.transform(self.X['jet'][:,ii,0].reshape(-1, 1)).flatten()
            self.X['jet'][:,ii,1] = scaler_eta.transform(self.X['jet'][:,ii,1].reshape(-1, 1)).flatten()
            self.X['jet'][:,ii,3] = scaler_mass.transform(self.X['jet'][:,ii,3].reshape(-1, 1)).flatten()
            #self.X['jet'][:,ii,5] = scaler_pt_log.transform(self.X['jet'][:,ii,5].reshape(-1, 1)).flatten()
            self._Dphi[:,ii] = scaler_dphi.transform(self._Dphi[:,ii].reshape(-1, 1)).flatten()

        if scalers is None:
            return scalers_dict
        else:
            return scalers

    def normalize_all(self, scalers=None):
        scalers_event = self.normalize_eventlevel_data(scalers)
        scalers_jet = self.normalize_jetlevel_data(scalers) 
        return {**scalers_event, **scalers_jet}
    
        
    def scale_eventlevel_data(self,dic):
        # event phi
        logging.info('Rescaling ev_phi')
        new_ev_phi = (self._ev_phi.reshape(-1, 1) - dic['scaler_phi']['mean'])/np.sqrt(dic['scaler_phi']['var'])
        self._ev_phi = new_ev_phi.flatten()
        # event eta
        logging.info('Rescaling ev_eta')
        new_ev_eta = (self._ev_eta.reshape(-1, 1) - dic['scaler_eta']['mean'])/np.sqrt(dic['scaler_eta']['var'])
        self._ev_eta = new_ev_eta.flatten()
        # event met
        logging.info('Rescaling ev_met')
        new_ev_met = (self._ev_met.reshape(-1, 1) - dic['scaler_met']['mean'])/np.sqrt(dic['scaler_met']['var'])
        self._ev_met = new_ev_met.flatten()
        # event MT2
        logging.info('Rescaling ev_MT2')
        new_ev_MT2 = (self._ev_MT2.reshape(-1, 1) - dic['scaler_MT2']['mean'])/np.sqrt(dic['scaler_MT2']['var'])
        self._ev_MT2 = new_ev_MT2.flatten()
        # event ht
        logging.info('Rescaling ev_ht')
        new_ev_ht = (self._ev_ht.reshape(-1, 1) - dic['scaler_ht']['mean'])/np.sqrt(dic['scaler_ht']['var'])
        self._ev_ht = new_ev_ht.flatten()
        # event mass
        logging.info('Rescaling ev_m')
        new_ev_m = (self._ev_m.reshape(-1, 1) - dic['scaler_m']['mean'])/np.sqrt(dic['scaler_m']['var'])
        self._ev_m = new_ev_m.flatten()


    def scale_jetlevel_data(self, dic):
        for ii in range(0, 4):
            logging.info(f'Rescaling jet{ii+1}')
            new_jetpt = (self.X['jet'][:,ii,0] - dic[f'scaler_jet{ii+1}PT']['mean'])/np.sqrt(dic[f'scaler_jet{ii+1}PT']['var'])
            self.X['jet'][:,ii,0] = new_jetpt.flatten()
            
            new_jeteta = (self.X['jet'][:,ii,1] - dic[f'scaler_jet{ii+1}ETA']['mean'])/np.sqrt(dic[f'scaler_jet{ii+1}ETA']['var'])
            self.X['jet'][:,ii,1] = new_jeteta.flatten()
            
            new_jetmass = (self.X['jet'][:,ii,3] - dic[f'scaler_jet{ii+1}MASS']['mean'])/np.sqrt(dic[f'scaler_jet{ii+1}MASS']['var'])
            self.X['jet'][:,ii,3] = new_jetmass.flatten()
            
            #new_jetpt_log = (self.X['jet'][:,ii,5] - dic[f'scaler_jet{ii+1}PT_log']['mean'])/np.sqrt(dic[f'scaler_jet{ii+1}PT_log']['var'])
            #self.X['jet'][:,ii,5] = new_jetpt_log.flatten()

            new_jetDphi = (self._Dphi[:,ii] - dic[f'scaler_jet{ii+1}DPHI']['mean'])/np.sqrt(dic[f'scaler_jet{ii+1}DPHI']['var'])
            self._Dphi[:,ii] = new_jetDphi.flatten()

            
    def scale_data(self,dic):
        self.scale_eventlevel_data(dic)
        self.scale_jetlevel_data(dic)
