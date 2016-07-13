# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 17:18:15 2016

@author: joostbloom
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import time
import datetime
import json

def quick_assess_features(feature_list, data, labels=['Duplicates','Original']):

    for f in feature_list:
        
        ddup = data[f][data['isDuplicate']==1]
        dori = data[f][data['isDuplicate']==0]
        
        newfigure("%s analyzed" % f)
        plt.hist(ddup, weights = np.zeros_like(ddup) + 1. / len(ddup), bins=30,alpha=0.5)
        plt.hist(dori, weights = np.zeros_like(dori) + 1. / len(dori), bins=30,alpha=0.5)
        plt.legend(labels)
        plt.grid()
        plt.xlabel('Value ' + f)
        
        print('Comparing {} ({} items)'.format(f, len(data)))
        print('   <dup>: {:.2f}'.format(np.mean(ddup)))
        print('   <ori>: {:.2f}'.format(np.mean(dori)))
        print('   rel: {:.2f}'.format(np.mean(ddup)/np.mean(dori)))
        
def show_random_items(data, features_to_show, label):
    d = data[data.isDuplicate==label]
    
    while True:
        itemID = int(random.uniform(0, len(d)))
        
        item = d.iloc[itemID]
        
        print("Showing item: {}".format(itemID))
        for f in features_to_show:
            print("{}:".format(f))
            if isinstance(item[f], list):
                print "[" + ", ".join(item[f]) + "]"
            elif isinstance(item[f], dict):
                for k,v in enumerate(item[f]):
                    print "{}: {}=".format(k,v), item[f][v]
            else:
                print(item[f])
        
        i = raw_input("Press enter to show next (or Enter 'q' to quit): ")
        
        if i=='q':
            break
        
    print("Done")


def start(description, ms_per_item=0, nitems=None):
    start_time = time.time()
    if nitems is None:
        print("Starting {}...".format(description)),
    else:
        now = datetime.datetime.now()
        eta = (now + datetime.timedelta(0,ms_per_item*nitems/1000)).strftime("%Y-%m-%d %H:%M:%S")
        print("Starting {}...(eta: {})".format(description,eta))
    return start_time
        

def report(starttime, description=None, nitems=None):
    delta = time.time()-starttime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    itemStr = ""
    if nitems is not None: itemStr = " (for {} items, {:.2f}ms per item)".format(nitems, delta/nitems*1000)
    
    # Add space to make in look prettier
    if description is None : description = ""
    if description is not None: description += " "
    
    print("{}done at {}. It took {:.2f}s{}.".format(description, timestamp, delta, itemStr)) 
    
def adjustfile_path(filepath, extraString='_sample'):
    filename = filepath[ (filepath.rfind('/')+1) : filepath.rfind('.') ]
    path = filepath[0:(filepath.rfind('/')+1)]
    
    return path + filename + extraString + '.csv'
    
def newfigure(title, width=650, height=500):
    plt.figure()
    mngr = plt.get_current_fig_manager()
    try:
        # Crashed on command line
        mngr.window.setGeometry(50,100,width, height)
    except:
        pass
    plt.title(title)
    
    return plt
def jsontodict(str_json):
    if len(str_json)==0:
        return {}
    else:
        return json.loads(str_json)