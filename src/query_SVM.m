load('variable/gist.mat');
load('variable/dataset.mat');

label = [];
for i=1:size(files,1)
	str = strsplit(files(i).name,'/');
	label = [label; str(3)];
end

[GroundTruth, Predicted] = svm_classify(gist,label);
save('variable/SVM.mat','GroundTruth','Predicted');

label = unique(GroundTruth);

cnt=[];		%total number of times Predicted
for i=1:size(label,1)
	cnt = [cnt; sum(Predicted==label(i))];
end

actual = [];
for i=1:size(label,1)		%count of many actually present
	actual = [actual; sum(GroundTruth==label(i))];
end

correct = zeros(21,1);

X=find(GroundTruth==Predicted);		%count of correctly Predicted
for i=1:size(X,1)
	index=find(label==GroundTruth(X(i)));
	correct(index)=correct(index)+1;
end

precision = nansum(correct./cnt)/21;

accuracy = nansum(correct./actual)/21;