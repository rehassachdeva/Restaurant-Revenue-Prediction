function [GroundTruthLabel,PredictedLabel] = svm_classify(X,Y,k)
%X : rows being files, columns being vector size
%Y : label vector for each file.
%k : number of bins in cross validation

    Y = categorical(Y);
    classOrder = unique(Y);
    %rng(1); 
    t = templateSVM('Standardize',1);
    CVMdl = fitcecoc(X,Y,'Kfold',k,'Learners',t,'ClassNames',classOrder);
    GroundTruthLabel = [];
    PredictedLabel = [];
    for i=1:k
        CMdl = CVMdl.Trained{i};           
        testInds = test(CVMdl.Partition,i);  
        XTest = X(testInds,:);
        labels = predict(CMdl,XTest);
        GroundTruthLabel = [GroundTruthLabel;YTest];
        PredictedLabel = [PredictedLabel;labels];
    end
end
