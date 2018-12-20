%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clustering and segmentation of time series (including with regime
% changes) by mixture of gaussian Hidden Markov Models Regression (MixFHMMR) and the EM algorithm; i.e functional data
% clustering and segmentation
%
%
%
%
% by Faicel Chamroukhi, 2009
%
%% Please cite the following references for this code
% 
% @InProceedings{Chamroukhi-IJCNN-2011,
%   author = {F. Chamroukhi and A. Sam\'e  and P. Aknin and G. Govaert},
%   title = {Model-based clustering with Hidden Markov Model regression for time series with regime changes},
%   Booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN), IEEE},
%   Pages = {2814--2821},
%   Adress = {San Jose, California, USA},
%   year = {2011},
%   month = {Jul-Aug},
%   url = {https://chamroukhi.com/papers/Chamroukhi-ijcnn-2011.pdf}
% }
% 
% @PhdThesis{Chamroukhi_PhD_2010,
% author = {Chamroukhi, F.},
% title = {Hidden process regression for curve modeling, classification and tracking},
% school = {Universit\'e de Technologie de Compi\`egne},
% month = {13 december},
% year = {2010},
% type = {Ph.D. Thesis},
% url ={https://chamroukhi.com/papers/FChamroukhi-Thesis.pdf}
% }
%
% @article{Chamroukhi-FDA-2018,
% 	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
% 	Author = {Faicel Chamroukhi and Hien D. Nguyen},
% 	Note = {DOI: 10.1002/widm.1298.},
% 	Volume = {},
% 	Title = {Model-Based Clustering and Classification of Functional Data},
% 	Year = {2019},
% 	Month = {to appear},
% 	url =  {https://chamroukhi.com/papers/MBCC-FDA.pdf}
% 	}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all;
clc;

%% simulated time series
load simulated_data.mat

Y; % time series


%% Model specification
K = 3;% number of clusters
R = 3;% number of regimes/states
p = 2;% degree of the polynomial regressors
  

%%
% variance_type = 'common';
variance_type = 'free';
ordered_sates = 1;% binary
total_EM_tries = 1;
max_iter_EM = 1000;
init_kmeans = 1;
threshold = 1e-6;
verbose = 1; 

%%
mixFHMMR =  learn_MixFHMMR_EM(Y, K, R, p, ...
    variance_type, ordered_sates, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose);
 
%% 
show_MixFHMMR_results(Y, mixFHMMR)
