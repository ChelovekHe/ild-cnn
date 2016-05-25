# coding: utf-8

# # ILD CNN prediction part
# ## this is run on the full patient data
#     the model is rebuilt
#     the weights are loaded from ILD_CNN_model_weights
import g
import os
import numpy as np
import cPickle as pickle

import ild_helpers as H
import cnn_model as CNN4


args         = H.parse_args()                          
train_params = {
     'do' : float(args.do) if args.do else 0.5,        
     'a'  : float(args.a) if args.a else 0.3,          # Conv Layers LeakyReLU alpha param [if alpha set to 0 LeakyReLU is equivalent with ReLU]
     'k'  : int(args.k) if args.k else 4,              # Feature maps k multiplier
     's'  : float(args.s) if args.s else 1,            # Input Image rescale factor
     'pf' : float(args.pf) if args.pf else 1,          # Percentage of the pooling layer: [0,1]
     'pt' : args.pt if args.pt else 'Avg',             # Pooling type: Avg, Max
     'fp' : args.fp if args.fp else 'proportional',    # Feature maps policy: proportional, static
     'cl' : int(args.cl) if args.cl else 5,            # Number of Convolutional Layers
     'opt': args.opt if args.opt else 'Adam',          # Optimizer: SGD, Adagrad, Adam
     'obj': args.obj if args.obj else 'ce',            # Minimization Objective: mse, ce
     'patience' : args.pat if args.pat else 5,         # Patience parameter for early stoping
     'tolerance': args.tol if args.tol else 1.005,     # Tolerance parameter for early stoping [default: 1.005, checks if > 0.5%]
     'res_alias': args.csv if args.csv else 'res'      # csv results filename alias
}


#print listcwd

#predict_dir = os.path.join(cwdtop, namedirtop)
model = H.load_model()

model.compile(optimizer='Adam', loss=CNN4.get_Obj(train_params['obj']))
#patient_list=dir_list = os.walk(predict_dir).next()[1]
#print (patient_list)

for f in g.patient_list:

    print ('predict work on: ',f)
    
    patient_dir_s = os.path.join(g.path_patient,f)
#    print patient_dir_s
    patient_dir_pkl= os.path.join(patient_dir_s, g.picklefile)
#    print patient_dir_pkl
    patient_pred_file =os.path.join( patient_dir_pkl,g.Xprepkl)
    X_predict = pickle.load( open( patient_pred_file, "rb" ) )
#    print X_predict
    # adding a singleton dimension and rescale to [0,1]
    X_predict = np.asarray(np.expand_dims(X_predict,1))/float(255)

    # predict and store  classification and probabilities 
    classes = model.predict_classes(X_predict, batch_size=10)
    proba = model.predict_proba(X_predict, batch_size=10)
    # store  classification and probabilities 
    xfc=os.path.join( patient_dir_pkl,g.predicted_classes)
    xfproba=os.path.join( patient_dir_pkl,g.predicted_proba)
    pickle.dump(classes, open( xfc, "wb" ))
    pickle.dump(proba, open( xfproba, "wb" ))

