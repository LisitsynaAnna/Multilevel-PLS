function out  = sk_multilevel_pls_adapted(X,Y,subj,ncomp,scaling,nfold,nrepeats,nperm)

    out.original_data_X = X;
    out.original_data_Y = Y;
    out.original_data_subj = subj;

    [n,p] = size(X);
    usubj = unique(subj);
    s = length(usubj);
    
    % split-up variation ( X = Xm + Xb + Xw )
    
    Xm = repmat(mean(X),n,1);
    Ym = repmat(mean(Y),n,1);
    
    Xc = X - Xm;
    Yc = Y - Ym;
    
    Xb = zeros(n,p);
    for i = 1:s
        Xb(subj == usubj(i),:) = repmat(mean(Xc(subj == usubj(i),:)), sum(subj == usubj(i)), 1);
    end
    
    Xw = Xc - Xb;
    
    out.Xm = Xm;
    out.Xc = Xc;
    out.Xb = Xb;
    out.Xw = Xw;
    
    out.Ym = Ym;
    out.Yc = Yc;
    
    if scaling == true
        Xbstd = repmat(std(Xb),n,1);
        Ycstd = repmat(std(Yc),n,1);
        Xbsc = Xb./Xbstd;
        Ycsc = Yc./Ycstd;
    else
        Xbstd = 1;
        Ycstd = 1;
        Xbsc = Xb/Xbstd;
        Ycsc = Yc/Ycstd;
    end
    
    out.Xbstd = Xbstd;
    out.Xbsc = Xbsc;
    
    out.Ycstd = Ycstd;
    out.Ycsc = Ycsc;
    
    [Px,Py,Tx,Ty,beta] = plsregress(Xbsc,Ycsc,ncomp);
    
    out.Px_train = Px;
    out.Tx_train = Tx;
    out.Py_train = Py;
    out.Ty_train = Ty;
    out.beta_train = beta;
    out.pred_train = ([ones(n,1), out.Xbsc]*out.beta_train).*out.Ycstd + out.Ym;    
    out.error_train = 100*sum(Y ~= ((out.pred_train > 0) - (out.pred_train < 0)))/n;
    out.sq_error_train = sum((Y-out.pred_train).^2)/n;
    out.sq_error_mod_train = sum((Y-(2./(1+exp(-2*out.pred_train))-1)).^2)/n;
    [tp,fp] = roc(Y,out.pred_train);
    out.auc_train = auroc(tp,fp);
    out.R2 = 1 - out.sq_error_train/(sum(Y'*Y)/n);
    
    cvrepeats = cell(nrepeats,5);
    cv_error_train = zeros(nrepeats,1);
    cv_error_test = zeros(nrepeats,1);
    cv_sq_error_train = zeros(nrepeats,1);
    cv_sq_error_mod_train = zeros(nrepeats,1);
    cv_sq_error_test = zeros(nrepeats,1);
    cv_sq_error_mod_test = zeros(nrepeats,1);
    
    for i = 1:nrepeats
        
        roc_curve_input = zeros(n,2);
        roc_counter = 1;
        cvsets = cell(nfold,6);
        temp_usubj = usubj;
        for j = 1:nfold

            if length(temp_usubj) >= ceil(s/nfold)
                cvsets{j,1} = datasample(temp_usubj,ceil(s/nfold),'Replace',false);
            else
                cvsets{j,1} = temp_usubj;
            end
            temp_usubj(ismember(temp_usubj,cvsets{j,1})) = [];
            cvsets{j,2} = usubj(~ismember(usubj,cvsets{j,1}));

            cv_pls = sk_multilevel_pls_cv_adapted(X,Y,subj,ncomp,scaling,cvsets(j,:));
            cvsets{j,3} = cv_pls.error_train;
            cvsets{j,4} = cv_pls.pred_train;
            cvsets{j,5} = cv_pls.error_cv;
            cvsets{j,6} = cv_pls.pred_test;

            cv_error_train(i,1) = cv_error_train(i,1) + cv_pls.error_train;
            cv_sq_error_train(i,1) = cv_sq_error_train(i,1) + cv_pls.sq_error_train;
            cv_sq_error_mod_train(i,1) = cv_sq_error_mod_train(i,1) + cv_pls.sq_error_mod_train;
            cv_error_test(i,1) = cv_error_test(i,1) + cv_pls.error_cv;
            cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
            cv_sq_error_mod_test(i,1) = cv_sq_error_mod_test(i,1) + cv_pls.sq_error_mod_cv;
            roc_curve_input(roc_counter:roc_counter+2*length(cvsets{j,1})-1,:) = [cv_pls.Y_test, cv_pls.pred_test];
            roc_counter = roc_counter + 2*length(cvsets{j,1});
            
        end
        cvrepeats{i,1} = cvsets;
        cv_error_train(i,1) = cv_error_train(i,1)/nfold;
        cv_sq_error_train(i,1) = cv_sq_error_train(i,1)/nfold;
        cv_sq_error_mod_train(i,1) = cv_sq_error_mod_train(i,1)/nfold;
        cv_error_test(i,1) = 100*cv_error_test(i,1)/n;
        cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/n;
        cv_sq_error_mod_test(i,1) = cv_sq_error_mod_test(i,1)/n;
        cvrepeats{i,2} = cv_error_test(i,1);
        cvrepeats{i,3} = cv_sq_error_test(i,1);
        cvrepeats{i,4} = cv_sq_error_mod_test(i,1);
        [tp,fp] = roc(roc_curve_input(:,1),roc_curve_input(:,2));
        cvrepeats{i,5} = auroc(tp,fp);
        
    end
    out.cvsets = cvrepeats;
    out.cv_error_train = mean(cv_error_train);
    out.cv_sq_error_train = mean(cv_sq_error_train);
    out.cv_sq_error_mod_train = mean(cv_sq_error_mod_train);
    out.cv_error_test = mean(cv_error_test);
    out.cv_sq_error_test = mean(cv_sq_error_test);
    out.cv_sq_error_mod_test = mean(cv_sq_error_mod_test);
    out.cv_auc = mean([cvrepeats{i,5}]);
    out.Q2 = 1 - out.cv_sq_error_test/(sum(Y'*Y)/n);

    if nperm > 0
    
        out.perm_cv_error = zeros(nperm,1);
        out.perm_Q2 = zeros(nperm,1);
        
        for k = 1:nperm
           
            clc;
            disp(k);
            
            indx = datasample(1:n,n,'Replace',false);
            Xperm = X(indx,:);
            Yperm = Y(indx,:);
  
            cv_error_test = zeros(nrepeats,1);
            cv_sq_error_test = zeros(nrepeats,1);
            
            for i = 1:nrepeats
        
                cvsets = cell(nfold,6);
                temp_usubj = usubj;
                for j = 1:nfold

                    if length(temp_usubj) >= ceil(s/nfold)
                        cvsets{j,1} = datasample(temp_usubj,ceil(s/nfold),'Replace',false);
                    else
                        cvsets{j,1} = temp_usubj;
                    end
                    temp_usubj(ismember(temp_usubj,cvsets{j,1})) = [];
                    cvsets{j,2} = usubj(~ismember(usubj,cvsets{j,1}));
            
                    cv_pls = sk_multilevel_pls_cv_adapted(Xperm,Yperm,subj,ncomp,scaling,cvsets(j,:));
                    cvsets{j,3} = cv_pls.error_train;
                    cvsets{j,4} = cv_pls.pred_train;
                    cvsets{j,5} = cv_pls.error_cv;
                    cvsets{j,6} = cv_pls.pred_test;

                    cv_error_test(i,1) = cv_error_test(i,1) + cv_pls.error_cv;
                    cv_sq_error_test(i,1) = cv_sq_error_test(i,1) + cv_pls.sq_error_cv;
            
                end

                cv_error_test(i,1) = 100*cv_error_test(i,1)/n;
                cv_sq_error_test(i,1) = cv_sq_error_test(i,1)/n;
        
            end
            
            out.perm_cv_error(k,1) = mean(cv_error_test);
            out.perm_Q2(k,1) = 1 - mean(cv_sq_error_test)/(sum(Y'*Y)/n);
            
        end
        
    end
    
end