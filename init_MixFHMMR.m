function param = init_MixFHMMR(data, G, K, Phi, variance_type, ordered_states, init_kmeans, try_algo)
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%%%%%%
D = data;
[n m]=size(D);

% % 1. Initialization of cluster weights
param.w_g=1/G*ones(G,1);
%Initialization of the model parameters for each cluster
if init_kmeans
    max_iter_kmeans = 400;
    n_tries_kmeans = 20;
    verbose_kmeans = 0;
    res_kmeans = myKmeans(D,G,n_tries_kmeans,max_iter_kmeans,verbose_kmeans);
    for g=1:G
        Xg = D(res_kmeans.klas==g ,:); %if kmeans
        param_init =  init_hmm_regression(Xg, K, Phi, ordered_states, variance_type, try_algo);
        
        % 3. Initialisation de la matrice des transitions        
        param.A_g(:,:,g)  =  param_init.trans_mat;
        if ordered_states
            param.mask = param_init.mask;
        end
        % 2. Initialisation de \pi^g
        param.pi_g(:,g) = param_init.initial_prob;%[1;zeros(K-1,1)];               
        % 4. Initialisation des coeffecients de regression et des variances.
        param.beta_gk(:,:,g) = param_init.betak;
        if strcmp(variance_type,'common')
            param.sigma_g(g) = param_init.sigma;
        else
            param.sigma_gk(:,g) = param_init.sigmak;
        end
    end
else
    ind = randperm(n);
    for g=1:G
        if g<G
            Xg = D(ind((g-1)*round(n/G) +1 : g*round(n/G)),:);
        else
            Xg = D(ind((g-1)*round(n/G) +1 : end),:);
        end
        param_init =  init_hmm_regression(Xg, K, Phi, ordered_states, variance_type, nb_try);
        % 3. Initialisation de la matrice des transitions        
        param.A_g(:,:,g)  =  param_init.trans_mat;
        if ordered_states
            param.mask = param_init.mask;
        end
        % 2. Initialisation de \pi^g
        param.pi_g(:,g) = param_init.initial_prob;%[1;zeros(K-1,1)];               
        % 4. Initialisation des coeffecients de regression et des variances.
        param.beta_gk(:,:,g) = param_init.betak;
        if strcmp(variance_type,'common')
            param.sigma_g(g) = param_init.sigma;
        else
            param.sigma_gk(:,g) = param_init.sigmak;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param =  init_hmm_regression(data, K, Phi, ordered_states, variance_type, try_algo)
% init_hmm_regression estime les paramètres initiaux d'un modèle de regression
% à processus markovien cache où la loi conditionnelle des observations est une gaussienne
%
% Entrees :
%       
%        data  = n sequences each sequence is of m points 
%        signaux les observations sont monodimentionnelles)
%        K : nbre d'états (classes) cachés
%        X :  matruce de r�gression
%
% Sorties :
%
%         param : parametres initiaux du modele. structure
%         contenant les champs: para: structrure with the fields:
%         * le HMM initial          
%         1. initial_prob (k) = Pr(Z(1) = k) avec k=1,...,K. loi initiale de z.
%         2. trans_mat(\ell,k) = Pr(z(i)=k | z(i-1)=\ell) : matrice des transitions
%         *         
%         3.betak : le vecteur parametre de regression associe a la classe k.
%         vecteur colonne de dim [(p+1)x1]
%         4. sigmak(k) = variance de x(i) sachant z(i)=k; sigmak(j) =
%         sigma^2_k.
% 
% Faicel Chamroukhi, Novembre 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% 1. Initialization of the HMM parameters
if ordered_states
    % Initialisation en tenant compte de la contrainte:
    % Initialisation de la matrice des transitions
    mask = eye(K);%mask d'ordre 1
    % mask = eye(K).*rand(K,K);%initialisation al�atoire
    for k=1:K-1;
        ind = find(mask(k,:) ~= 0);
        mask(k,ind+1) = 1;
    end
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = [1;zeros(K-1,1)];
    param.trans_mat = normalize(mask,2);%
    param.mask = mask;
 else
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = [1;zeros(K-1,1)];%1/K*ones(K,1);
    param.trans_mat = mk_stochastic(rand(K));
end


% % 2.  Initialisation of regression coefficients and variances

regression_param = init_regression_param(data, K, Phi, variance_type, try_algo);

param.betak = regression_param.betak;
if strcmp(variance_type,'common')
    param.sigma = regression_param.sigma;
else
    param.sigmak = regression_param.sigmak;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = init_regression_param(data, K, X,variance_type, try_algo)
 
% init_regression_param initialize the Regresssion model with Hidden Logistic Process
%
% X: regression matrix
%
%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n m] = size(data);
% p = size(phi,2)-1;
Y = data;

if strcmp(variance_type,'common')
    s=0;
end 
%

if try_algo ==1
    %decoupage de l'echantillon (signal) en K segments
    zi = round(m/K)-1;
    for k=1:K
        i = (k-1)*zi+1;
        j = k*zi;
        
        Yij = Y(:,i:j);
        Yij = reshape(Yij',[],1);
        
        X_ij=repmat(X(i:j,:),n,1);
           
        bk = inv(X_ij'*X_ij)*X_ij'*Yij; 
        
        param.betak(:,k) = bk;
        
        if strcmp(variance_type,'common')
            s=s+ sum((Yij-X_ij*bk).^2);
           param.sigma = s/(n*m);
         else
             mk = j-i+1 ; 
             z = Yij-X_ij*bk;
             sk = z'*z/(n*mk); 
             param.sigmak(k) = sk;
         end
     end
 else % initialisation aléatoire
     Lmin= round(m/(K+1));%nbr pts min dans un segments
     tk_init = zeros(1,K+1);
     tk_init(1) = 0;         
     K_1=K;
     for k = 2:K
         K_1 = K_1-1;
         temp = tk_init(k-1)+Lmin:m-K_1*Lmin;
         ind = randperm(length(temp));
         tk_init(k)= temp(ind(1));                      
     end
     tk_init(K+1) = m;
     %model.tk_init = tk_init;
     for k=1:K
         i = tk_init(k)+1;
         j = tk_init(k+1);
         Yij = Y(:,i:j);
         Yij = reshape(Yij',[],1);
        
        X_ij=repmat(X(i:j,:),n,1);
           
        bk = inv(X_ij'*X_ij)*X_ij'*Yij; 
        param.betak(:,k) = bk;
        
        if strcmp(variance_type,'common')
           s=s+ sum((Yij-X_ij*bk).^2);
           param.sigma = s/(n*m);
        else
             mk = j-i+1 ;%length(Yij);
             z = Yij-X_ij*bk;
             sk = z'*z/(n*mk); 
             param.sigmak(k) = sk;
            %param.sigmak(k) = var(Yij);
         end
     end
 end
    