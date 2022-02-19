from flask import Flask, render_template, request,jsonify
from werkzeug.utils import secure_filename
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.conversion.log import converter as log_converter
import pandas as pd
import pygraphviz as pgv
from datetime import datetime
from datetime import timedelta
import scipy.stats as stats
from scipy.stats import norm
import os
import networkx
import json
import time

log = 'None!'

app = Flask(__name__)
app.debug = True
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    #get filename
    file_name = request.files['file']
    #import log, find start-and endtime of the uploaded event log
    global log, startTime, endTime
    log = xes_importer.apply(file_name.filename)
    startTime = (min([event["time:timestamp"] for trace in log for event in trace])).strftime("%Y-%m-%d %H:%M:%S")
    endTime = (max([event["time:timestamp"] for trace in log for event in trace]) + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
    return render_template('app.html', startTime =startTime, endTime = endTime)

@app.route('/test/', methods=['GET', 'POST'])
def main_interface():

    if request.method == "POST":

        global log
        #get request from the user
        req = request.get_json()
        print(req)

        #save the request features in variables
        global startTime, endTime, sna_edges, ap_supp, m_supp, wd_edges
        startDate = datetime(*(time.strptime(req["startDate"],"%Y-%m-%dT%H:%M:%S.%f%z")[0:6])) #requested startDate
        endDate = datetime(*(time.strptime(req["endDate"],"%Y-%m-%dT%H:%M:%S.%f%z")[0:6])) #requested endDate
        sna_edges = req["sna_edges"] #requested maximal nr of edges for Social Network Analysis
        wd_edges = req["wd_edges"] #requested maximal nr of edges for Work Distribution
        ap_supp = req["ap_supp"] #requested minimal support for candidate constraints for apriori algorithm
        if ap_supp != '':
            ap_supp = float(ap_supp)
        else:
            ap_supp = 0.8 #default value
        m_supp = req["m_supp"] #requested minimal support for declare constraints
        if m_supp != '':
            m_supp = float(m_supp)
        else:
            m_supp = 0.9 #default value

        global filenameid #used for the graph filenames
        filenameid = sna_edges + wd_edges + str(ap_supp) + str(m_supp)


        global color, edgethickness, font
        # functions(for the color of nodes, edge thickness, and font color )
        #sna network
        def sna_color(x):
            if x >= 0.0 and x < 0.05 :
                color = '#ebf0fa'
                edgethickness = '2'
                font = 'black'
            elif x >= 0.05 and x < 0.15:
                color = '#c2d1f0'
                edgethickness = '2.5'
                font = 'black'
            elif x >= 0.15 and x < 0.35:
                color = '#99b3e6'
                edgethickness = '3'
                font = 'black'
            elif x >= 0.35 and x < 0.5:
                color = '#7094db'
                edgethickness = '3.5'
                font = 'black'
            elif x >= 0.5 and x < 0.65:
                color = '#3366cc'
                edgethickness = '4'
                font = 'white'
            elif x >= 0.65 and x < 0.85:
                color = '#2e5cb8'
                edgethickness = '4.5'
                font = 'white'
            elif x >= 0.85 and x < 0.95:
                color = '#2952a3'
                edgethickness = '5'
                font = 'white'
            elif x >= 0.95 and x <= 1:
                color = '#24478f'
                edgethickness = '5.5'
                font = 'white'
            return color, edgethickness,font
        #control-flow graph
        def cf_color(x):
            if x >= 0.0 and x < 0.05 :
                color = '#ffeecc'
                edgethickness = '2'
                font = 'black'
            elif x >= 0.05 and x < 0.15:
                color = '#ffe6b3'
                edgethickness = '2.5'
                font = 'black'
            elif x >= 0.15 and x < 0.35:
                color = '#ffdd99'
                edgethickness = '3'
                font = 'black'
            elif x >= 0.35 and x < 0.5:
                color = '#ffd480'
                edgethickness = '3.5'
                font = 'black'
            elif x >= 0.5 and x < 0.65:
                color = '#ffcc66'
                edgethickness = '4'
                font = 'white'
            elif x >= 0.65 and x < 0.85:
                color = '#ffc34d'
                edgethickness = '4.5'
                font = 'white'
            elif x >= 0.85 and x < 0.95:
                color = '#ffbb33'
                edgethickness = '5'
                font = 'white'
            elif x >= 0.95 and x <= 1:
                color = '#ffb31a'
                edgethickness = '6'
                font = 'white'
            return color, edgethickness,font
        #performance graph
        def performace_color(x):
            if x >= 0.0 and x < 0.05 :
                color = '#fad1d1'
                edgethickness = '2'
                font = 'black'
            elif x >= 0.05 and x < 0.15:
                color = '#f6a2a2'
                edgethickness = '2.5'
                font = 'black'
            elif x >= 0.15 and x < 0.35:
                color = '#f17474'
                edgethickness = '3'
                font = 'black'
            elif x >= 0.35 and x < 0.5:
                color = '#ef5d5d'
                edgethickness = '3.5'
                font = 'black'
            elif x >= 0.5 and x < 0.65:
                color = '#ea2e2e'
                edgethickness = '4'
                font = 'white'
            elif x >= 0.65 and x < 0.85:
                color = '#d11515'
                edgethickness = '4.5'
                font = 'white'
            elif x >= 0.85 and x < 0.95:
                color = '#8b0e0e'
                edgethickness = '5'
                font = 'white'
            elif x >= 0.95 and x <= 1:
                color = '#5d0909'
                edgethickness = '6'
                font = 'white'
            return color, edgethickness,font

        ############################ Concept Drift Detection ########################################
        #subdivide log in time intervals
        def subdivide_log(log, start, end, windowsize):
            logs = []
            n = (end - start).days//(windowsize)
            #print(n)
            for i in range(0,n):
                logs.append(timestamp_filter.apply_events(log,
                                                          (start+ timedelta(days=i*windowsize)).strftime("%Y-%m-%d %H:%M:%S"),
                                                          (start+ timedelta(days=(i+1)*windowsize)).strftime("%Y-%m-%d %H:%M:%S")))
            return logs

        sT = datetime(*(time.strptime(startTime,"%Y-%m-%d %H:%M:%S")[0:3]))
        eT = datetime(*(time.strptime(endTime,"%Y-%m-%d %H:%M:%S")[0:3]))
        logs = subdivide_log(log,sT-timedelta(seconds=86400),eT+timedelta(seconds=86400), 7)

        def apply_feature_extraction(logs, features):
            feature_vectors = [[] for i in range(0,len(logs))]
            for feature in features:
                results = []
                if feature == "duration":
                    #results = the duration of each event
                    results = extract_event_durations(logs)
                if feature == 'workload':
                    #results = the workload of each participants
                    results = extract_workload(logs)
                #more features
                for i in range(0,len(results)):
                    for result in results[i]:
                        feature_vectors[i].append(result)
            #set non existent features to zero
            feature_names__ = []
            for i in range(0,len(logs)):
                for j in range(0,len(feature_vectors[i])):
                    feature_names__.append(feature_vectors[i][j][0])

            feature_names = list(set(feature_names__))

            feature_lists = []
            for i in range(0,len(feature_names)):
                feature_list = []
                for j in range(0,len(logs)):
                    existing_features = [k[0] for k in feature_vectors[j]]
                    if feature_names[i] in existing_features:
                        #find index
                        idx = existing_features.index(feature_names[i])
                        feature_list.append(feature_vectors[j][idx][1])
                    else:
                        feature_list.append(0)
                feature_lists.append(feature_list)

            return feature_names, np.asarray(feature_lists).transpose()

        ##### Event Duration #####

        from pm4py.objects.log.util import interval_lifecycle
        import numpy as np
        def extract_event_durations(logs):
            results = []
            for log in logs:
                duration_lists = {}
                enriched_log = interval_lifecycle.assign_lead_cycle_time(log)
                for case in enriched_log:
                    for event in case:
                        if event["concept:name"] not in duration_lists.keys():
                            duration_lists[event["concept:name"]] = []
                        duration_lists[event["concept:name"]].append(event["@@duration"])
                log_results = [(key, sum(duration_lists[key])/len(duration_lists[key])) for key in duration_lists.keys()]
                results.append(log_results)
            return results

        ##### Workload #####

        def extract_workload(logs):
            results = []
            for log in logs:
                workload = {}
                workload["total"] = 0
                for trace in log:
                    for event in trace:
                        if 'org:resource' in event.keys():
                            res = event['org:resource']
                            if not res in workload.keys():
                                workload[res] = 0
                            workload[res]+=1
                            workload["total"] += 1
                log_results = [('Resource_'+str(res),workload[res]) for res in workload.keys()]
                results.append(log_results)
            return results


        from sklearn.decomposition import PCA
        def pca_reduction(features_np, dimensions, normalize = False, normalize_function = 'max'):
            if normalize:
                row_sums = features_np.sum(axis=0) + 0.0001
                if normalize_function == 'max':
                    row_sums = features_np.max(axis=0) + 0.0001
                new_matrix = features_np / row_sums[np.newaxis, : ]
                features_np = new_matrix
            tmp_features = features_np
            if dimensions == 'mle':
                if features_np.shape[1] > features_np.shape[0]:
                    pca = PCA(n_components = features_np.shape[0], svd_solver ="full")
                    pca.fit(features_np)
                    tmp_features = pca.transform(features_np)
            pca = PCA(n_components = dimensions, svd_solver ="full")
            pca.fit(tmp_features)
            reduced_features = pca.transform(tmp_features)

            if reduced_features.shape[1]==0:
                pca = PCA(n_components = 1, svd_solver ="full")
                pca.fit(tmp_features)
                reduced_features = pca.transform(tmp_features)
            return reduced_features

        primary_names, primary_features = apply_feature_extraction(logs, ["duration"])
        reduced_primary = pca_reduction(primary_features,
                                'mle',
                                normalize = True,
                                normalize_function="max")
        secondary_names, secondary_features = apply_feature_extraction(logs,["workload"])
        reduced_secondary = pca_reduction(secondary_features,
                                  'mle',
                                  normalize = True,
                                  normalize_function="max")

        import ruptures as rpt
        import matplotlib.pylab as plt

        # PELT (Pruned Exact Linear Time) Algorithm
        #change point detection technique for multivariate time serie

        def rpt_pelt(series, pen =3):

            algo = rpt.Pelt(model="rbf",min_size=1,jump=1).fit(series)
            result = algo.predict(pen=pen)
            #display
            #rpt.display(series, result)
            #plt.show()
            return result[:-1]

        pen_primary = 1
        cp_1 = rpt_pelt(reduced_primary, pen = pen_primary)

        pen_secondary = 1
        cp_2 = rpt_pelt(reduced_secondary, pen = pen_secondary)


        # identify the the weeks and calendar week when concept drift occurred in the workload of the users
        workload_change_points = []
        if len(cp_2) != 0:
            for i in cp_2:
                #print(cp_2,i)
                if len([event["time:timestamp"] for trace in logs[i] for event in trace])!=0:
                    mi = min([event["time:timestamp"] for trace in logs[i] for event in trace])
                    kwi  = mi.strftime("%W")
                    li = mi.strftime("%Y-%m-%d")
                    ma = max([event["time:timestamp"] for trace in logs[i] for event in trace])
                    kwa = ma.strftime("%W")
                    la = ma.strftime("%Y-%m-%d")
                    if kwi != kwa:
                        kw = ' KW: '+ kwi+ ' - ' + kwa
                    else:
                        kw = ' KW: '+ kwi
                    workload_change_points.append([li + ' - ' + la + kw])
        else:
                workload_change_points.append('No Change Point Detection')

        # identify the the weeks and calendar week when concept drift occurred in the cycle time of the activities
        duration_change_points = []
        if len(cp_1) != 0:
            for i in cp_1:
                if len([event["time:timestamp"] for trace in logs[i] for event in trace])!=0:
                    mi = min([event["time:timestamp"] for trace in logs[i] for event in trace])
                    kwi  = mi.strftime("%W")
                    li =mi.strftime("%Y-%m-%d")
                    ma = max([event["time:timestamp"] for trace in logs[i] for event in trace])
                    la =ma.strftime("%Y-%m-%d")
                    kwa = ma.strftime("%W")
                    if kwi != kwa:
                        kw = ' KW: '+ kwi+ ' - ' + kwa
                    else:
                        kw = ' KW: '+ kwi
                    duration_change_points.append([li + ' - ' + la+kw])
        else:
            duration_change_points.append('No Change Point Detection')


        ##filter log within the chosen timestamp (based on the request of the user)
        fLog = timestamp_filter.filter_traces_contained(log,startDate, endDate)

        # convert log to dataframe
        df = log_converter.apply(fLog, variant=log_converter.TO_DATA_FRAME)
        df = df.rename(columns = {'concept:name': 'Task',
                                  'case:concept:name':'Case ID',
                                  'lifecycle:transition':'EventType',
                                  'org:resource':'User',
                                  'time:timestamp':'Timestamp'}, inplace = False)

        df = df.drop_duplicates()
        #print(df)
        #if there is data in the chosen interval the analysis, we continue with the calculations
        if len(df) != 0:

            df['User']= df['User'].astype(str)
            #convert the log to dictionary
            dLog = {}
            for i in df["Case ID"].unique():
                dLog[i] = [(df["Task"][j], df["EventType"][j], df["User"][j], df["Timestamp"][j]) for j in df[df["Case ID"]==i].index]

            #filter df by only having only completed events
            df_comp = df[df.EventType.str.startswith('COMP')]
            dLog_comp = {} #converted filterd log to dictionary
            for i in df_comp["Case ID"].unique():
                dLog_comp[i] = [(df_comp["Task"][j], df_comp["EventType"][j], df_comp["User"][j], df_comp["Timestamp"][j]) for j in df_comp[df_comp["Case ID"]==i].index]


            ############################################################################################
            ##########################SOCIAL NETWORK ANALYSIS###########################################

            ###### HANDOVER OF WORK  ######
            handoverOfWork = dict()
            for caseid in dLog_comp:
                for i in range(0, len(dLog_comp[caseid])-1):
                    user_i =dLog_comp[caseid][i][2]
                    user_j =dLog_comp[caseid][i+1][2]
                    if user_i not in handoverOfWork:
                        handoverOfWork[user_i] = dict()
                    if user_j not in handoverOfWork[user_i]:
                        handoverOfWork[user_i][user_j] = 0
                    handoverOfWork[user_i][user_j] += 1
            print(handoverOfWork)
            ##### Workload PER USER #####
            workloadPerUser = dict()
            for caseid in dLog_comp:
                for i in range(0, len(dLog_comp[caseid])):
                    user_i =dLog_comp[caseid][i][2]
                    if user_i not in workloadPerUser:
                        workloadPerUser[user_i] = 0
                    workloadPerUser[user_i] += 1
            print(workloadPerUser)


            ##### SOCIAL NETWORK GRAPH ######
            G = pgv.AGraph(strict=False, directed=True)
            G.graph_attr['rankdir'] = 'LR'
            G.node_attr['shape'] = 'circle'
            sna_nodes = pd.DataFrame()
            actperUserMIN = min(workloadPerUser.values()) #the min nr of activites
            actperUserMAX = max(workloadPerUser.values()) #the max nr of activites
            for user_i in workloadPerUser:
                differenceRatio = (workloadPerUser[user_i] - actperUserMIN)/(actperUserMAX - actperUserMIN)
                graph_nodes = pd.Series([user_i, workloadPerUser[user_i], differenceRatio])
                sna_nodes = sna_nodes.append(graph_nodes, ignore_index=True)
            sna_nodes.columns =['User', 'NumberOfActivities', 'DifferenceRatio']
            import scipy.stats as stats
            from scipy.stats import norm
            sna_nodes['cdf'] = norm.cdf(stats.zscore(sna_nodes['DifferenceRatio']), loc=0, scale=1)
            for i in range (0, len(sna_nodes)):
                color, edgethickness, font = sna_color(sna_nodes['cdf'][i])
                text = sna_nodes['User'][i] + '\n(' + str(int(sna_nodes['NumberOfActivities'][i])) + ')'
                G.add_node(sna_nodes['User'][i], label= text, style='filled', fillcolor=color, fontcolor=font, width='1.5', fixedsize=True, color =color,fontsize='20', fontname = 'Helvetica' )

            values = [handoverOfWork[user_i][user_j] for user_i in handoverOfWork for user_j in handoverOfWork[user_i]]
            min_values = min(values)
            max_values = max(values)
            #r:  the min nr of handover of work s.t. the graph has the requested nr of edgges
            snaMinEdgeWeight = 0
            if sna_edges != "":
                snaMinEdgeWeight = []
                for k in values:
                    count = sum(i >= k for i in values)
                    if count <= int(sna_edges):
                        snaMinEdgeWeight.append(k)
                snaMinEdgeWeight = min(snaMinEdgeWeight)
            else:
                snaMinEdgeWeight=0

            sna_edge = pd.DataFrame()
            for user_i in handoverOfWork:
                for user_j in handoverOfWork[user_i]:
                    x = handoverOfWork[user_i][user_j]
                    if x >= snaMinEdgeWeight:
                        differenceRatio = (x - min_values)/(max_values - min_values)
                        graph_edges = pd.Series([user_i,user_j, x, differenceRatio])
                        sna_edge = sna_edge.append(graph_edges, ignore_index=True)
            sna_edge.columns =['User','NextUser','NumberOfActivities','DifferenceRatio']
            sna_edge['cdf'] = norm.cdf(stats.zscore(sna_edge['DifferenceRatio']), loc=0, scale=1)
            for i in range (0, len(sna_edge)):
                color,edgethickness,font= sna_color(sna_edge['cdf'][i])
                G.add_edge(sna_edge['User'][i], sna_edge['NextUser'][i], label=int(sna_edge['NumberOfActivities'][i]), penwidth=edgethickness)

            filename = 'static/Network  ' + str(startDate)+' -- '+str(endDate)+' ' + filenameid+'.svg'
            G.draw(filename, prog='circo')

            ############# SNA Metrics #############

            #build a networkx graph to calculate the complexity measures
            import networkx as nx
            dff = pd.DataFrame()
            row = []
            for user_i in handoverOfWork:
                for user_j, numberOfActivities in handoverOfWork[user_i].items():
                    row.append([user_i,user_j,numberOfActivities])
            dff = dff.append(row, ignore_index=True)
            dff.columns =['Source', 'Target', 'Weight']
            N = nx.from_pandas_edgelist(dff, 'Source', 'Target', 'Weight', create_using=nx.DiGraph())

            nrOfEdges =  N.number_of_edges()
            nrOfNodes =  N.number_of_nodes()
            density =  round(nx.density(N),3)
            clustCoe = round(nx.average_clustering(N),3)
            transitivity =  round(nx.transitivity(N),3)

            #Degree
            degree = dict(N.degree())
            minDegreeKV = min(zip(degree.values(), degree.keys()))[1] + " ("+ str(round(min(zip(degree.values(), degree.keys()))[0],3)) + ")"
            maxDegreeKV = max(zip(degree.values(), degree.keys()))[1] + " ("+ str(round(max(zip(degree.values(), degree.keys()))[0],3)) + ")"
            #Betweeness Centrality
            btwCentrality = nx.betweenness_centrality(N)
            minbKV = min(zip(btwCentrality.values(), btwCentrality.keys()))[1] + " ("+ str(round(min(zip(btwCentrality.values(), btwCentrality.keys()))[0],3)) + ")"
            maxbKV = max(zip(btwCentrality.values(), btwCentrality.keys()))[1] + " ("+ str(round(max(zip(btwCentrality.values(), btwCentrality.keys()))[0],3)) + ")"

            #Closeness Centrality
            clsCentrality = nx.closeness_centrality(N)
            mincKV = min(zip(clsCentrality.values(), clsCentrality.keys()))[1] + " ("+ str(round(min(zip(clsCentrality.values(), clsCentrality.keys()))[0],3)) + ")"
            maxcKV = max(zip(clsCentrality.values(), clsCentrality.keys()))[1] + " ("+ str(round(max(zip(clsCentrality.values(), clsCentrality.keys()))[0],3)) + ")"

            # Clustering
            clustering = nx.clustering(N)
            minClcKV = min(zip(clustering.values(), clustering.keys()))[1] + " ("+ str(round(min(zip(clustering.values(), clustering.keys()))[0],3)) + ")"
            maxClcKV = max(zip(clustering.values(), clustering.keys()))[1] + " ("+ str(round(max(zip(clustering.values(), clustering.keys()))[0],3)) + ")"

            #Degree Centrality
            centrality = nx.degree_centrality(N)
            maxdegCenV = max(zip(centrality.values(), centrality.keys()))[1] + ' ('+ str(round(max(zip(centrality.values(), centrality.keys()))[0],3))+ ")"
            mindegCenV = min(zip(centrality.values(), centrality.keys()))[1] + ' ('+ str(round(min(zip(centrality.values(), centrality.keys()))[0],3))+ ")"


            ##################################################################################################
            ############################## Performance Analysis ##############################################

            #Control flow graph
            # find which activity comes after which one
            control_flow = dict()
            for caseid in dLog_comp:
                for i in range(0, len(dLog_comp[caseid])-1):
                    activity_i = dLog_comp[caseid][i][0]
                    activity_j = dLog_comp[caseid][i+1][0]
                    if activity_i not in control_flow:
                        control_flow[activity_i] = dict()
                    if activity_j not in control_flow[activity_i]:
                        control_flow[activity_i][activity_j] = 0
                    control_flow[activity_i][activity_j] += 1
            #print("control_flows")
            print(control_flow)

            #how often one activity was executed
            total_activites = dict()
            for caseid in dLog_comp:
                for i in range(0, len(dLog_comp[caseid])):
                    activity_i = dLog_comp[caseid][i][0]

                    if activity_i not in total_activites:
                        total_activites[activity_i] = 0
                    total_activites[activity_i] += 1
            print(total_activites)
            P = pgv.AGraph(strict=False, directed=True)
            P.graph_attr['rankdir'] = 'L'
            P.node_attr['shape'] = 'box'

            pa_nodes = pd.DataFrame()
            totalActMIN = min(total_activites.values())
            totalActMAX = max(total_activites.values())

            for activity_i in total_activites:
                differenceRatio = (total_activites[activity_i] - totalActMIN)/(totalActMAX - totalActMIN)
                graph_nodes = pd.Series([activity_i, total_activites[activity_i], differenceRatio])
                pa_nodes = pa_nodes.append(graph_nodes, ignore_index=True)
            pa_nodes.columns =['Activity', 'NumberOfActivities', 'DifferenceRatio']
            import scipy.stats as stats
            from scipy.stats import norm
            pa_nodes['cdf'] = norm.cdf(stats.zscore(pa_nodes['DifferenceRatio']), loc=0, scale=1)
            for i in range (0, len(pa_nodes)):
                color, edgethickness, font = cf_color(pa_nodes['cdf'][i])
                text = pa_nodes['Activity'][i] + '\n(' + str(int(pa_nodes['NumberOfActivities'][i])) + ')'
                #nodewidth
                if len(text)/8 < 2.5:
                    nodeWidth = 2.5
                else:
                    nodeWidth = len(text)/8
                P.add_node(pa_nodes['Activity'][i], label= text, style='filled', fillcolor=color, fontcolor=font, width = nodeWidth,height='0.8',fixedsize=True, color =color,fontsize='20', fontname = 'Helvetica' )

            values = [control_flow[ai][aj] for ai in control_flow for aj in control_flow[ai]]
            min_values = min(values)
            max_values = max(values)

            pa_edge = pd.DataFrame()
            for activity_i in control_flow:
                for activity_j in control_flow[activity_i]:
                    x = control_flow[activity_i][activity_j]
                    differenceRatio = (x - min_values)/(max_values - min_values)
                    graph_edges = pd.Series([activity_i,activity_j, x, differenceRatio])
                    pa_edge = pa_edge.append(graph_edges, ignore_index=True)
            pa_edge.columns =['Activity','NextActivity','NumberOfActivities','DifferenceRatio']
            pa_edge['cdf'] = norm.cdf(stats.zscore(pa_edge['DifferenceRatio']), loc=0, scale=1)
            for i in range (0, len(pa_edge)):
                color,edgethickness,font= cf_color(pa_edge['cdf'][i])
                P.add_edge(pa_edge['Activity'][i], pa_edge['NextActivity'][i], label=int(pa_edge['NumberOfActivities'][i]), penwidth=edgethickness)
            #draw graph
            process = 'static/Process  ' + str(startDate)+' -- '+str(endDate)+' ' + filenameid+'.svg'
            P.draw(process, prog='dot')




            ############ KPI #############

            ###### WAITING TIME ###########
            # average waiting time considering also previous activity
            waitingTime = dict()
            # total average waiting time  without considering previous acitivty
            totalWT = dict()

            for caseid in dLog:
                for i in range(0, len(dLog[caseid])-1):
                    (ai, ei, ui , ti) = dLog[caseid][i]
                    for j in range(i+1, len(dLog[caseid])):
                        (aj, ej, uj,  tj) = dLog[caseid][j]

                        if df['EventType'].str.contains('SCHEDULE').any() == True:
                            if ai == aj and ei == 'SCHEDULE' and ej == 'START':
                                for n in range(j+1, len(dLog[caseid])):
                                    (an, en, un,  tn) = dLog[caseid][j+1]
                                    if aj != an:
                                        if ai not in waitingTime:
                                            waitingTime[ai] = dict()
                                            totalWT[ai] = []
                                        if an not in waitingTime[ai]:
                                            waitingTime[ai][an] = []
                                        waitingTime[ai][an].append(tj-ti)
                                        totalWT[ai].append(tj-ti)
                                        break
                        else:
                            if ei == 'COMPLETE' and ej == 'START':
                                if ai not in waitingTime:
                                    waitingTime[ai] = dict()
                                    totalWT[ai] = []
                                if aj not in waitingTime[ai]:
                                    waitingTime[ai][aj] = []
                                waitingTime[ai][aj].append(tj-ti)
                                totalWT[ai].append(tj-ti)
                                break


            for ai in sorted(waitingTime.keys()):
                for aj in sorted(waitingTime[ai].keys()):
                    sum_td = sum(waitingTime[ai][aj], timedelta(0))
                    count_td = len(waitingTime[ai][aj])
                    avg_td = sum_td/count_td
                    avg_td -= timedelta(microseconds=avg_td.microseconds)
                    waitingTime[ai][aj] = avg_td

            for ai in sorted(totalWT.keys()):
                sum_td = sum(totalWT[ai], timedelta(0))
                count_td = len(totalWT[ai])
                avg_td = sum_td/count_td
                avg_td -= timedelta(microseconds=avg_td.microseconds)
                totalWT[ai] = str(avg_td)

            # find minimunm and maximimum waiting time
            maxWTV,maxWT_Task,aprevT = max(((v,k,l) for l in waitingTime.keys() for k,v in waitingTime[l].items()))
            maxWTK = maxWT_Task + ' from ' + aprevT
            maxWTKV = maxWTK + " - "+ str(maxWTV).split('.')[0]

            minWTV,minWT_Task,iprevT = min(((v,k,l) for l in waitingTime.keys() for k,v in waitingTime[l].items()))
            minWTK = minWT_Task + ' from ' + iprevT
            minWTKV = minWTK + " - "+ str(minWTV).split('.')[0]


            ######## PROCESSING TIME #######

            processingTime = dict()
            for caseid in dLog:
                for i in range(0, len(dLog[caseid])-1):
                    (ai, ei, ui , ti) = dLog[caseid][i]
                    for j in range(i+1, len(dLog[caseid])):
                        (aj, ej, uj,  tj) = dLog[caseid][j]
                        if ai == aj and ei == 'START' and ej == 'COMPLETE':
                            if ai not in processingTime:
                                processingTime[ai] = dict()
                            if aj not in processingTime[ai]:
                                processingTime[ai][aj] = []
                            processingTime[ai][aj].append(tj-ti)
                            break

            for ai in sorted(processingTime.keys()):
                for aj in sorted(processingTime[ai].keys()):
                    sum_td = sum(processingTime[ai][aj], timedelta(0))
                    count_td = len(processingTime[ai][aj])
                    avg_td = sum_td/count_td
                    avg_td -= timedelta(microseconds=avg_td.microseconds)
                    processingTime[ai] = avg_td

            #minimum and maximimum processing time
            minPTKV = str(min(zip(processingTime.values(), processingTime.keys()))[1] + " - "+ str(min(zip(processingTime.values(), processingTime.keys()))[0])).split('.')[0]
            maxPTKV = str(max(zip(processingTime.values(), processingTime.keys()))[1] + " - "+ str(max(zip(processingTime.values(), processingTime.keys()))[0])).split('.')[0]

            ####### THROUGHPUT TIME
            tp = df.groupby('Case ID')['Timestamp'].agg(['max','min'])
            throughput = str((tp['max']-tp['min']).mean()).split('.')[0]


            from dateutil.parser import parse
            import math

            # Generate performance Graph
            PP = pgv.AGraph(strict=False, directed=True)
            PP.graph_attr['rankdir'] = 'L'
            PP.node_attr['shape'] = 'box'

            ptMIN = min(processingTime.values()).total_seconds()
            ptMAX = max(processingTime.values()).total_seconds()
            pp_nodes = pd.DataFrame()
            for activity in processingTime:
                differenceRatio = (processingTime[activity].total_seconds() - ptMIN)/(ptMAX - ptMIN)
                graph_nodes = pd.Series([activity, processingTime[activity], differenceRatio])
                pp_nodes = pp_nodes.append(graph_nodes, ignore_index=True)

            pp_nodes.columns =['Activities', 'ProcessingTime', 'DifferenceRatio']
            import scipy.stats as stats
            from scipy.stats import norm
            pp_nodes['cdf'] = norm.cdf(stats.zscore(pp_nodes['DifferenceRatio']), loc=0, scale=1)

            for i in range (0, len(pp_nodes)):
                color, edgethickness, font = performace_color(pp_nodes['cdf'][i])
                text = pp_nodes['Activities'][i] + '\n(' + str(pp_nodes['ProcessingTime'][i]).split('.')[0] + ')'
                PP.add_node(pp_nodes['Activities'][i], label= text, style='filled',fillcolor=color, fontcolor=font, width=len(text)/10,height='0.8', fixedsize=True, color =color,fontsize='20', fontname = 'Helvetica' )

            values = [waitingTime[activity_i][activity_j] for activity_i in waitingTime for activity_j in waitingTime[activity_i]]

            wtMIN = min(values).total_seconds()
            wtMAX = max(values).total_seconds()
            pp_edge = pd.DataFrame()

            for activity_i in waitingTime:
                for activity_j in waitingTime[activity_i]:
                    x = waitingTime[activity_i][activity_j]
                    x1 = x.total_seconds()
                    differenceRatio = round((x1-wtMIN)/(wtMAX-wtMIN),3)
                    graph_edges = pd.Series([activity_i,activity_j,x, differenceRatio])
                    pp_edge = pp_edge.append(graph_edges, ignore_index=True)

            pp_edge.columns =['Activity', 'NextActivity', 'Waiting Time', 'DifferenzRatio']
            import scipy.stats as stats
            from scipy.stats import norm

            pp_edge['cdf'] = norm.cdf(stats.zscore(pp_edge['DifferenzRatio']), loc=0, scale=1)
            for i in range (0, len(pp_edge)):
                color,edgethickness,font = performace_color(pp_edge['cdf'][i])
                PP.add_edge(pp_edge['Activity'][i], pp_edge['NextActivity'][i], label=str(pp_edge['Waiting Time'][i]).split('.')[0], penwidth=edgethickness)

            processPerformance = 'static/ProcessPerformance  ' + str(startDate)+' -- '+str(endDate)+' ' + filenameid+'.svg'
            PP.draw(processPerformance, prog='dot')

            for ai in sorted(waitingTime.keys()):
                for aj in sorted(waitingTime[ai].keys()):
                    waitingTime[ai][aj] = str(waitingTime[ai][aj])
            for ai in sorted(processingTime.keys()):
                processingTime[ai] = str(processingTime[ai])
######################################################################################################################

            ####################### WORK DISTRIBUTION ###################

            ##### THE ACTIVITIES PERFORMED BY EACH USER ######
            user_Activity = dict()
            for caseid in dLog_comp:
                for i in range(0, len(dLog_comp[caseid])):
                    activity_i = dLog_comp[caseid][i][0]
                    user_i = dLog_comp[caseid][i][2]
                    if user_i not in user_Activity:
                        user_Activity[user_i] = dict()
                    if activity_i not in user_Activity[user_i]:
                        user_Activity[user_i][activity_i] = 0
                    user_Activity[user_i][activity_i] += 1
            user_Activity

            all_activities = []
            # generate work distribution graph
            K = pgv.AGraph(strict=False, directed=False)
            for user_i in user_Activity:
                font = 'black'
                K.add_node(user_i, label=user_i, style='filled', shape='circle', fillcolor='#adc2eb', color ='#adc2eb', fontcolor=font,width='1.5',height='0.8',fontsize='20', fontname = 'Helvetica')
            for user_i in user_Activity:
                for activity_i in user_Activity[user_i]:
                    all_activities.append(activity_i)
            all_activities = list(dict.fromkeys(all_activities))
            for activity_i in range(0, len(all_activities)):
                K.add_node(all_activities[activity_i], label=all_activities[activity_i], style='filled', shape='box', fillcolor='#ffd480',color = '#ffd480', fontcolor=font,width=len(text)/8,height='0.8',fontsize='20', fontname = 'Helvetica')

            values = [user_Activity[user_i][activity_i] for user_i in user_Activity for activity_i in user_Activity[user_i]]

            actMIN = min(values)
            actMAX = max(values)
            #wd:  calculate the minimum weight of the edges in the wd graph
            wdMinEdgeweight = 0
            if wd_edges != "":
                wdMinEdgeweight = []
                for k in values:
                    count = sum(i >= k for i in values)
                    if count <= int(wd_edges):
                        wdMinEdgeweight.append(k)
                wdMinEdgeweight = min(wdMinEdgeweight)
            else:
                wdMinEdgeweight = 0

            for user_i in user_Activity:
                for activity_i in user_Activity[user_i]:
                    x = user_Activity[user_i][activity_i]
                    if x >=wdMinEdgeweight:
                        if actMAX-actMIN != 0:
                            K.add_edge(user_i, activity_i, label=x, penwidth=3+(x-actMIN)/(actMAX-actMIN))
                        else:
                            K.add_edge(user_i, activity_i, label=x, penwidth=3+(x-actMIN))

            #draw graph
            actxuser = 'static/actxuser  ' + str(startDate)+' -- '+str(endDate)+' ' + filenameid+'.svg'
            K.draw(actxuser, prog='dot')

########################################################################################################################


            ############### DECLARE CONSTRAINTS ###############
            global existence_constraints,all_constraints,negativeConstraints
            existence_constraints =[]
            all_constraints = []
            negativeConstraints = []

            ##########  existence templates ##########
            def init(a,dLog_comp):
                global existence_constraints
                con = dict()
                for caseid in dLog_comp:
                    if dLog_comp[caseid][0] == a:
                        con[caseid] = 1
                supp_con = sum(con.values()) / len(dLog_comp)
                if supp_con >= m_supp:
                    constraint_init = ['Init', a, supp_con]
                    existence_constraints.append(constraint_init)
                    return existence_constraints


            def end(a,dLog_comp):
                global existence_constraints
                con = dict()
                for caseid in dLog_comp:
                    if dLog_comp[caseid][-1] == a:
                        con[caseid] = 1
                supp_con = sum(con.values()) / len(dLog_comp)
                if supp_con >= m_supp:
                    constraint_end = ['End', a, supp_con]
                    existence_constraints.append(constraint_end)
                    return existence_constraints


            def existence(a, n,dLog_comp):
                global existence_constraints
                occurred = 0
                con = dict()
                for caseid in dLog_comp:
                    for i in range(0, len(dLog_comp[caseid])):
                        if dLog_comp[caseid][i] == a:
                            occurred += 1
                            con[caseid] = occurred
                        else:
                            occurred += 0
                            con[caseid] = occurred
                    occurred = 0
                d = dict((k, v) for k, v in con.items() if v >= n)
                supp_con = len(d) / len(dLog_comp)
                if supp_con >= m_supp:
                    constraint_existence = ['Existence', str(int(n)) + ', ' + a, supp_con]
                    existence_constraints.append(constraint_existence)
                    return existence_constraints


            def absence(a, m,dLog_comp):
                global existence_constraints
                con = dict()
                occurred = 0
                nrOfTraces = len(dLog_comp)
                for caseid in dLog_comp:
                    for i in range(0, len(dLog_comp[caseid])):
                        if dLog_comp[caseid][i] == a:
                            occurred += 1
                            con[caseid] = occurred
                        else:
                            occurred += 0
                            con[caseid] = occurred
                    occurred = 0
                d = dict((k, v) for k, v in con.items() if v <= m)
                supp_con = len(d) / nrOfTraces
                if supp_con >= m_supp:
                    constraint_absence = ['Absence', str(int(m)) + ', ' + a, supp_con]
                    existence_constraints.append(constraint_absence)
                    return existence_constraints

            ########## relation templates ##########
            def responded_Existence(a, b, dLog_comp):
                global respondedExistence
                global constraint_respExist
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, b]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_respExist = ['RESPONDED_EXISTENCE', a, b, supp_con]
                        respondedExistence = True
                        #print('respondedExistence - ', respondedExistence)
                    else:
                        respondedExistence = False
                        #print('respondedExistence - ', respondedExistence)


            def response(a, b, dLog_comp):
                global resp
                global constraint_resp
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][i + 1: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, b]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_resp = ['RESPONSE', a, b, supp_con]
                        resp = True
                        #print('response -', resp)
                    else:
                        resp = False
                        #print('response -', resp)


            def precedence(a, b, dLog_comp):
                global prec
                global constraint_prec
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if b in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                if a in dLog_comp[caseid][0: i]:
                                    dff = dff.append([[caseid, e, a]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_prec = ['PRECEDENCE', a, b, supp_con]
                        prec = True
                        #print('precedence -', prec)
                    else:
                        constraint_prec = ''
                        prec = False
                        #print('precedence -', prec)


            def alternateResponse(a, b, dLog_comp):
                global alternate_Response
                global constraint_altRes
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][i + 1: len(dLog_comp[caseid])]:
                                    for j in range(i + 1, len(dLog_comp[caseid])):
                                        f = dLog_comp[caseid][j]
                                        if f == a:
                                            dff = dff.append([[caseid, e, '0']])
                                            break
                                        if f == b:
                                            dff = dff.append([[caseid, e, b]])
                                            break
                                else:
                                    dff = dff.append([[caseid, a, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_altRes = ['ALTERNATE_RESPONSE', a, b, supp_con]
                        alternate_Response = True
                        #print('alternate_Response - ', alternate_Response)
                    else:
                        alternate_Response = False
                        #print('alternate_Response- ', alternate_Response)


            def alternatePrecedence(a, b, dLog_comp):
                global alternate_Precedence
                global constraint_altPrec
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if b in dLog_comp[caseid]:
                        for i in range(1, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                if a in dLog_comp[caseid][0: i]:
                                    for j in range(i - 1, -1, -1):
                                        f = dLog_comp[caseid][j]
                                        if f == b:
                                            dff = dff.append([[caseid, e, '0']])
                                            break
                                        if f == a:
                                            dff = dff.append([[caseid, e, a]])
                                            break
                                else:
                                    dff = dff.append([[caseid, b, '0']])
                        if dLog_comp[caseid][0] == b:
                            dff = dff.append([[caseid, b, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    # print(dff,nrofCaseid_novio,supp_con)
                    if supp_con >= m_supp:
                        constraint_altPrec = ['ALTERNATE_PRECEDENCE', a, b, supp_con]
                        alternate_Precedence = True
                        #print('alternate_Precedence - ', alternate_Precedence)
                    else:
                        alternate_Precedence = False
                        #print('alternate_Precedence - ', alternate_Precedence)


            def chainResponse(a, b, dLog_comp):
                global chain_Response
                global constraint_chainRes
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                f = dLog_comp[caseid][i + 1]
                                if f == b:
                                    dff = dff.append([[caseid, e, f]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_chainRes = ['CHAIN_RESPONSE', a, b, supp_con]
                        chain_Response = True
                        #print('chain_Response - ', chain_Response)
                    else:
                        constraint_chainRes = ''
                        chain_Response = False
                        #print('chain_Response - ', chain_Response)


            def chainPrecedence(a, b, dLog_comp):
                global chain_Precedence
                global constraint_chainPrec
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if b in dLog_comp[caseid]:
                        for i in range(1, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                f = dLog_comp[caseid][i - 1]
                                if f == a:
                                    dff = dff.append([[caseid, e, a]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][0] == b:
                            dff = dff.append([[caseid, b, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    nrOfTraces = len(dLog_comp)
                    supp_con = nrofCaseid_novio / nrOfTraces
                    if supp_con >= m_supp:
                        constraint_chainPrec = ['CHAIN_PRECEDENCE', a, b, supp_con]
                        chain_Precedence = True
                        #print('chain_Precedence -', chain_Precedence)
                    else:
                        chain_Precedence = False
                        #print('chain_Precedence -', chain_Precedence)

            ########## Mutual relation tmeplates ##########

            def chainSuccession(a, b, dLog_comp):
                global chain_Succession
                global constraint_chainSuc
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                f = dLog_comp[caseid][i + 1]
                                if f == b:
                                    dff = dff.append([[caseid, e, b]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    if b in dLog_comp[caseid]:
                        for i in range(1, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                f = dLog_comp[caseid][i - 1]
                                if f == a:
                                    dff = dff.append([[caseid, e, a]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][0] == b:
                            dff = dff.append([[caseid, b, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    # print(dff)
                    if supp_con >= m_supp:
                        constraint_chainSuc = ['CHAIN_SUCCESSION', a, b, supp_con]
                        chain_Succession = True
                        #print('CHAIN_SUCCESSION -', chain_Succession)
                    else:
                        chain_Succession = False
                        #print('CHAIN_SUCCESSION -', chain_Succession)


            def succession(a, b, dLog_comp):
                global suc
                global constraint_suc
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][i + 1: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, b]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    if b in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                if a in dLog_comp[caseid][0: i]:
                                    dff = dff.append([[caseid, e, a]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_suc = ['SUCCESSION', a, b, supp_con]
                        suc = True
                        #print('succession -', suc)
                    else:
                        constraint_prec = ''
                        suc = False
                        #print('succession -', suc)


            def alternateSuccession(a, b, dLog_comp):
                global alternate_Succession
                global constraint_altSuc
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][i + 1: len(dLog_comp[caseid])]:
                                    for j in range(i + 1, len(dLog_comp[caseid])):
                                        f = dLog_comp[caseid][j]
                                        if f == a:
                                            dff = dff.append([[caseid, e, '0']])
                                            break
                                        if f == b:
                                            dff = dff.append([[caseid, e, b]])
                                            break
                                else:
                                    dff = dff.append([[caseid, a, '0']])
                        if dLog_comp[caseid][-1] == a:
                            dff = dff.append([[caseid, a, '0']])
                    if b in dLog_comp[caseid]:
                        for i in range(1, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                if a in dLog_comp[caseid][0: i]:
                                    for j in range(i - 1, -1, -1):
                                        f = dLog_comp[caseid][j]
                                        if f == b:
                                            dff = dff.append([[caseid, e, '0']])
                                            break
                                        if f == a:
                                            dff = dff.append([[caseid, e, a]])
                                            break
                                else:
                                    dff = dff.append([[caseid, b, '0']])
                        if dLog_comp[caseid][0] == b:
                            dff = dff.append([[caseid, b, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_altSuc = ['ALTERNATE_SUCCESSION', a, b, supp_con]
                        alternate_Succession = True
                        #print('ALTERNATE_SUCCESSION - ', alternate_Succession)
                    else:
                        alternate_Succession = False
                        #print('ALTERNATE_SUCCESSION - ', alternate_Succession)

            def coExistence(a, b, dLog_comp):
                global coEx
                global constraint_coEx
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, b]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                    if b in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid])):
                            e = dLog_comp[caseid][i]
                            if e == b:
                                if a in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, a]])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_coEx = ['COEXISTENCE', a, b, supp_con]
                        coEx = True
                        #print('coExistence - ', coEx)
                    else:
                        coEx = False
                        #print('coExistence - ', coEx)


            #give constraint a value based on their hierarchy
            def getcode(constraint_name):
                code = 0
                if constraint_name == 'CHAIN_SUCCESSION':
                    code = 8
                elif constraint_name == 'CHAIN_RESPONSE' or constraint_name == 'CHAIN_PRECEDENCE':
                    code = 7
                elif constraint_name == 'ALTERNATE_SUCCESSION':
                    code = 6
                elif constraint_name == 'ALTERNATE_RESPONSE' or constraint_name == 'ALTERNATE_PRECEDENCE':
                    code = 5
                elif constraint_name == 'SUCCESSION':
                    code = 4
                elif constraint_name == 'RESPONSE' or constraint_name == 'PRECEDENCE':
                    code = 3
                elif constraint_name == 'COEXISTENCE':
                    code = 2
                elif constraint_name == 'RESPONDED_EXISTENCE' or constraint_name == 'RESPONDED_EXISTENCE':
                    code = 1
                elif constraint_name == 'NOT-COEXISTENCE':
                    code = 3
                elif constraint_name == 'NOT-SUCCESSION':
                    code = 2
                elif constraint_name == 'NOT_CHAIN_SUCCESSION':
                    code = 1
                else:
                    code = 0
                return code

            ##### find which mutual (relation) constraints is not violated ####
            def declareCon(a, b, dLog_comp):
                global constraint_chainSuc, constraint_altSuc, constraint_suc, constraint_coEx
                global constraint_chainRes, constraint_altRes, constraint_resp
                global constraint_chainPrec, constraint_altPrec, constraint_prec
                global constraint_notCoEx, constraint_notSucc, constraint_notChainSucc

                global chain_Succession, alternate_Succession, suc, coEx
                global chain_Response, alternate_Response, resp, respondedExistence, coEx
                global chain_Precedence, alternate_Precedence, prec, constraint_respExist
                global notCoEx, notSucc, notChainCoEx

                global all_constraints
                result = []

                chain_Succession = suc = coEx = False
                chain_Response = alternate_Response = resp = respondedExistence = coEx = False
                chain_Precedence = alternate_Precedence = prec = False
                notCoEx = notSucc = notChainCoEx = False

                constraint_chainSuc = constraint_altSuc = constraint_suc = constraint_coEx = []
                constraint_notCoEx = constraint_notSucc = constraint_notChainCoEx = []
                constraint_chainRes = constraint_altRes = constraint_resp = []
                constraint_chainPrec = constraint_altPrec = constraint_prec = constraint_respExist = []

                chainSuccession(a, b, dLog_comp)

                if chain_Succession == False:
                    chainResponse(a, b, dLog_comp)
                    chainPrecedence(a, b, dLog_comp)
                    if chain_Response == True and chain_Precedence == False:
                        alternateSuccession(a, b, dLog_comp)
                        if alternate_Succession == True:
                            result = constraint_altSuc
                        else:
                            succession(a, b, dLog_comp)
                            if suc == True:
                                result = constraint_suc
                            else:
                                coExistence(a, b, dLog_comp)
                                if coEx == True:
                                    result = constraint_coEx
                                else:
                                    result = constraint_chainRes

                    elif chain_Response == False and chain_Precedence == True:
                        alternateSuccession(a, b, dLog_comp)
                        if alternate_Succession == True:
                            result = constraint_altSuc
                        else:
                            succession(a, b, dLog_comp)
                            if suc == True:
                                result = constraint_suc
                            else:
                                coExistence(a, b, dLog_comp)
                                if coEx == True:
                                    result = constraint_coEx
                                else:
                                    result = constraint_chainPrec

                    elif chain_Response == False and chain_Precedence == False:
                        alternateSuccession(a, b, dLog_comp)
                        if alternate_Succession == True:
                            result = constraint_altSuc
                        else:
                            alternateResponse(a, b, dLog_comp)
                            alternatePrecedence(a, b, dLog_comp)

                            if alternate_Response == True and alternate_Precedence == False:
                                succession(a, b, dLog_comp)
                                if suc == True:
                                    result = constraint_suc
                                else:
                                    coExistence(a, b, dLog_comp)
                                    if coEx == True:
                                        result = constraint_coEx
                                    else:
                                        result = constraint_altRes
                            elif alternate_Response == False and alternate_Precedence == True:
                                succession(a, b, dLog_comp)
                                if suc == True:
                                    result = constraint_suc
                                else:
                                    coExistence(a, b, dLog_comp)
                                    if coEx == True:
                                        result = constraint_coEx
                                    else:
                                        result = constraint_altPrec
                            elif alternate_Response == False and alternate_Precedence == False:
                                succession(a, b, dLog_comp)
                                if suc == True:
                                    result = constraint_suc
                                else:
                                    response(a, b, dLog_comp)
                                    precedence(a, b, dLog_comp)
                                    if resp == True and prec == False:
                                        coExistence(a, b, dLog_comp)
                                        if coEx == True:
                                            result = constraint_coEx
                                        else:
                                            result = constraint_resp
                                    elif resp == False and prec == True:
                                        coExistence(a, b, dLog_comp)
                                        if coEx == True:
                                            result = constraint_coEx
                                        else:
                                            result = constraint_prec
                                    elif resp == False and prec == False:
                                        coExistence(a, b, dLog_comp)
                                        if coEx == True:
                                            result = constraint_coEx
                                        else:
                                            responded_Existence(a, b, dLog_comp)
                                            if respondedExistence == True:
                                                result = constraint_respExist
                                            else:
                                                responded_Existence(b, a, dLog_comp)
                                                if respondedExistence == True:
                                                    result = constraint_respExist
                else:
                    #print(constraint_chainSuc)
                    result = constraint_chainSuc
                return result

            ########### negative templates  ###########
            def notCoExistence(a, b, dLog_comp):
                global notCoEx
                global constraint_notCoEx
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    for i in range(0, len(dLog_comp[caseid])):
                        if any(c in dLog_comp[caseid] for c in (a, b)):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, '0']])
                                else:
                                    dff = dff.append([[caseid, e, '1']])
                            if e == b:
                                if a in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, '0']])
                                else:
                                    dff = dff.append([[caseid, e, '1']])
                        else:
                            dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_notCoEx = ['NOT-COEXISTENCE', a, b, supp_con]
                        notCoEx = True
                        #print('notCoExistence -', notCoEx)
                    else:
                        notCoEx = False
                        #print('notCoExistence -', notCoEx)


            def notSuccession(a, b, dLog_comp):
                global notSucc
                global constraint_notSucc
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    for i in range(0, len(dLog_comp[caseid]) - 1):
                        if any(c in dLog_comp[caseid] for c in (a, b)):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][i + 1: len(dLog_comp[caseid])]:
                                    dff = dff.append([[caseid, e, '0']])
                                else:
                                    dff = dff.append([[caseid, e, '1']])
                            if e == b:
                                if a in dLog_comp[caseid][0: i]:
                                    dff = dff.append([[caseid, e, '0']])
                                else:
                                    dff = dff.append([[caseid, e, '1']])
                        else:
                            dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_notSucc = ['NOT-SUCCESSION', a, b, supp_con]
                        notSucc = True
                        #print('notSuccession -', notSucc)
                    else:
                        notSucc = False
                        #print('notSuccession -', notSucc)


            def notChainSuccession(a, b, dLog_comp):
                global notChainSucc
                global constraint_notChainSucc
                dff = pd.DataFrame()
                for caseid in dLog_comp:
                    if a in dLog_comp[caseid]:
                        for i in range(0, len(dLog_comp[caseid]) - 1):
                            e = dLog_comp[caseid][i]
                            if e == a:
                                if b in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                    f = dLog_comp[caseid][i + 1]
                                    if f == b:
                                        dff = dff.append([[caseid, e, '0']])
                                    else:
                                        dff = dff.append([[caseid, e, '1']])
                                else:
                                    dff = dff.append([[caseid, e, '0']])
                        if dLog_comp[caseid][-1] == a:
                            if b in dLog_comp[caseid][0: len(dLog_comp[caseid])]:
                                dff = dff.append([[caseid, e, '1']])
                    else:
                        dff = dff.append([[caseid, '1', '1']])
                if len(dff) != 0:
                    dff.columns = ['Caseid', 'From', 'To']
                    dff = dff.groupby('Caseid').filter(lambda g: ~ (g.To == '0').any())
                    nrofCaseid_novio = dff['Caseid'].nunique()
                    supp_con = nrofCaseid_novio / len(dLog_comp)
                    if supp_con >= m_supp:
                        constraint_notChainSucc = ['NOT_CHAIN_SUCCESSION', a, b, supp_con]
                        notChainSucc = True
                        #print('notChainSuccession -', notChainSucc)
                    else:
                        notChainSucc = False
                        #print('notChainSuccession -', notChainSucc)

            #### find which negative constraint is not violated
            def negConstraints(a,b,dLog_comp):
                global notCoEx, notSucc, notChainSucc
                global constraint_notCoEx, constraint_notChainSucc, constraint_notSucc
                notCoEx =  notSucc = notChainSucc= False
                constraint_notCoEx = constraint_notChainSucc = constraint_notSucc = []
                negresult = []
                notCoExistence(a,b,dLog_comp)
                if notCoEx == True:
                    negresult = constraint_notCoEx
                else:
                    notSuccession(a,b,dLog_comp)
                    if notSucc == True:
                        negresult = constraint_notSucc
                    else:
                        notChainSuccession(a,b,dLog_comp)
                        if notChainSucc == True:
                            negresult = constraint_notChainSucc
                return negresult


            def comp_negativeconstraint(a, b, dLog_comp):
                first = negConstraints(a, b, dLog_comp)
                second = negConstraints(b, a, dLog_comp)
                if len(first) != 0 and len(second) != 0:
                    code1 = getcode(first[0])
                    code2 = getcode(second[0])
                    if code1 > code2:
                        negativeConstraints.append(first)
                    elif code1 < code2:
                        negativeConstraints.append(second)
                    elif code1 == code2:
                        negativeConstraints.append(first)
                        negativeConstraints.append(second)
                elif len(first) != 0 and len(second) == 0:
                    negativeConstraints.append(first)
                elif len(first) == 0 and len(second) != 0:
                    negativeConstraints.append(second)


            def comp_relationconstraint(a, b, dLog_comp):
                first = declareCon(a, b, dLog_comp)
                second = declareCon(b, a, dLog_comp)
                if len(first) != 0 and len(second) != 0:
                    code1 = getcode(first[0])
                    code2 = getcode(second[0])
                    if code1 > code2:
                        all_constraints.append(first)
                    elif code1 < code2:
                        all_constraints.append(second)
                    elif code1 == code2:
                        all_constraints.append(first)
                        all_constraints.append(second)
                elif len(first) != 0 and len(second) == 0:
                    all_constraints.append(first)
                elif len(first) == 0 and len(second) != 0:
                    all_constraints.append(second)

            #### find which activities/users can have constraints between them
            def declarefunction(declaretype):
                global all_constraints
                global existence_constraints
                global negativeConstraints
                all_constraints =[]
                existence_constraints = []
                negativeConstraints = []

                if declaretype == 'User':
                    listOfTraces = list(df_comp.groupby('Case ID')['User'].agg(list))
                    dLog_comp = {}
                    for i in df_comp["Case ID"].unique():
                        dLog_comp[i] = [(df_comp["User"][j]) for j in df_comp[df_comp["Case ID"]==i].index]
                elif declaretype == 'Activities':
                    listOfTraces = list(df_comp.groupby('Case ID')['Task'].agg(list))
                    dLog_comp = {}
                    for i in df_comp["Case ID"].unique():
                        dLog_comp[i] = [(df_comp["Task"][j]) for j in df_comp[df_comp["Case ID"]==i].index]



                ##### APRIORI ALGORITHM #####
                from mlxtend.frequent_patterns import apriori, association_rules
                from mlxtend.preprocessing import TransactionEncoder
                from statistics import mean

                a = TransactionEncoder()
                a_data = a.fit(listOfTraces).transform(listOfTraces)
                candidate_constraints = pd.DataFrame(a_data,columns=a.columns_)
                candidate_constraints = candidate_constraints.replace(False,0)
                candidate_constraints = apriori(candidate_constraints, min_support = ap_supp, max_len = 2, use_colnames = True, verbose = 1)

                candidate_constraints['length']=candidate_constraints['itemsets'].apply(lambda x: len(x))
                candidate_constraints['itemsets'] = candidate_constraints['itemsets'].apply(list)
                candidate_constraints


                for i in range(0, len(candidate_constraints)):
                    if candidate_constraints.iloc[i, 2] == 1:
                        a = candidate_constraints['itemsets'][i][0]
                        init(a,dLog_comp)
                        end(a,dLog_comp)
                        #existence(a, 1,dLog_comp)
                        #absence(a, 1,dLog_comp)
                    if candidate_constraints.iloc[i, 2] == 2:
                        a = candidate_constraints['itemsets'][i][0]
                        b = candidate_constraints['itemsets'][i][1]
                        print('-----', a, b, '----')
                        comp_relationconstraint(a,b,dLog_comp)
                        comp_negativeconstraint(a, b,dLog_comp)

                ### generate declate model
                if len(all_constraints) !=0:
                    all_constraints.sort(key=lambda x: x[3], reverse=True)
                    all_constraintsdf = pd.DataFrame(all_constraints)
                    all_constraintsdf.columns = ['name', 'from','to','support']
                    all_constraintsdf = all_constraintsdf.dropna()
                    print('all constraints',all_constraintsdf)
                    column_values = all_constraintsdf[["from", "to"]].values.ravel()
                    unique_values =  pd.unique(column_values)
                    unique_values = list(unique_values)
                    print(unique_values)
                    DC = pgv.AGraph(strict=False, directed=True)
                    DC.graph_attr['rankdir'] = 'L'
                    con_df=all_constraintsdf
                    con_dict = {}
                    for i in con_df["from"].unique():
                        con_dict[i] = [[con_df["to"][j], con_df["name"][j], con_df["support"][j]] for j in con_df[con_df["from"]==i].index]
                    #print(con_dict)

                    pr = ''
                    for ai in unique_values:
                        #print(ai)
                        text = ai
                        font = 'black'
                        if declaretype == 'User':
                            color = '#adc2eb'
                            DC.node_attr['shape'] = 'circle'
                            pr = 'circo'
                            width = '1.5'
                            height='0.8'
                        elif declaretype == 'Activities':
                            DC.node_attr['shape'] = 'box'
                            color = '#ffd480'
                            pr = 'dot'
                            width = '2.5'
                            height='0.8'
                        if len(text)/8 < 2.5:
                            nodeWidth = 2.5
                        else:
                            nodeWidth = len(text)/8
                        DC.add_node(ai, label=text, style='filled',fillcolor=color, color = color, fontcolor=font,width=nodeWidth,height = height, fixedsize=True,fontsize='20', fontname = 'Helvetica')
                    for ai in con_dict:
                        for i in con_dict[ai]:
                            aj = i[0]
                            x = i[1]
                            #print(ai,aj,x)
                            DC.add_edge(ai, aj, label=x)

                    declare = 'static/DeclareConstraints  ' + str(startDate)+' -- '+str(endDate)+ ' '+ declaretype +filenameid+ '.svg'
                    DC.draw(declare, prog=pr)
                    final_constraints = []
                    mutualRelationConstraints = []
                    for i in all_constraints:
                        mutualRelationConstraints.append([i[0]+'('+i[1]+', '+''+ i[2]+') = ' + str(round(i[3],3))])
                    for i in all_constraints:
                        final_constraints.append([i[0]+'('+i[1]+', '+''+ i[2]+') = ' + str(round(i[3],3))])
                    if len(negativeConstraints) !=0:
                        for i in negativeConstraints:
                            final_constraints.append([i[0]+'('+i[1]+', '+''+ i[2]+') = ' + str(round(i[3],3))])
                    if  len(existence_constraints) !=0:
                        for i in existence_constraints:
                            final_constraints.append([i[0]+'('+str(i[1])+') = ' + str(round(i[2],3))])
                else:
                    mutualRelationConstraints = ''
                    final_constraints = ''
                    declare = ''

                return mutualRelationConstraints, final_constraints, declare

            actmutualRelationConstraints, actFinalCon, actDeclare = declarefunction('Activities')
            usermutualRelationConstraints, userFinalCon, userDeclare = declarefunction('User')

            ### create json file
            name = 'metrics/Output  ' + str(startDate)+' -- '+str(endDate)+' ' + filenameid+'.json'
            with open(name, 'w') as f:
                json.dump({'Social Network Analysis':
                                                        {'NumberOfEdges:':nrOfEdges,
                                                        'NumberOfNodes:':nrOfNodes,
                                                        'Average Clustering Coefficient':clustCoe,
                                                        'Transitivity' : transitivity,
                                                        'Degree':degree,
                                                        'Betweenness Centrality' : btwCentrality,
                                                        'Closeness Centrality' : clsCentrality,
                                                        'Clustering Coefficient' : clustering,
                                                        'Degree Centrality':centrality,
                                                        'Duration Change Point Detection':duration_change_points
                                                        },
                           'Performance Analysis' : {
                                                        'Average Waiting Time' : totalWT,
                                                        'Average Waiting Time considering Previous Activity' : waitingTime,
                                                        'Average Processing Time' : processingTime,
                                                        'Average Throughput Time' : throughput
                                                        },
                            'Concept Drift' : { 'Workload Change Point Detection':  workload_change_points,
                                                'Cycle Time Change Point Detection': duration_change_points
                                                },
                           'User DECLARE Constraints': userFinalCon,
                           'Activities DECLARE Constraints':actFinalCon,

                           'Work Distribution': user_Activity

                           }, f, indent = 5)


                return jsonify({'nrOfEdges' : nrOfEdges,
                            'nrOfNodes' : nrOfNodes,
                            'density': density,
                            'transitivity': transitivity,
                            'clustCoe': clustCoe,
                             'minClcKV':minClcKV,
                            'maxClcKV':maxClcKV,
                            'minDegreeKV':minDegreeKV,
                            'maxDegreeKV':maxDegreeKV,
                            'minbKV' : minbKV,
                            'maxbKV' : maxbKV,
                            'minbKV' : minbKV,
                            'maxbKV' : maxbKV,
                            'mincKV' : mincKV,
                            'maxcKV' : maxcKV,
                            'clsCentrality' : clsCentrality,
                            'filename': filename,
                            'process':process,
                            'userdeclare':userDeclare,
                            'actdeclare':actDeclare,
                            'actxuser':actxuser,
                            'throughput':throughput,
                            'minWTKV':minWTKV,
                            'maxWTKV':maxWTKV,
                            'minPTKV':minPTKV,
                            'maxPTKV':maxPTKV,
                            'userdeccon':usermutualRelationConstraints[:15],
                            'actdeccon':actmutualRelationConstraints[:15],
                            'duration_change_points':duration_change_points,
                            'workload_change_points': workload_change_points,
                            'processPerformance' :processPerformance,
                            'maxdegCenV':maxdegCenV,
                            'mindegCenV':mindegCenV
                            })
        else:
            return jsonify({'p' : 'No data during this time. Choose another time interval.'})



@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')

    return response
if __name__ == '__main__':
    app.run(debug=True)
