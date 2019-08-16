clc
clear 
close all
vs_layer_dir = './layers_vs/';
allfiles=strsplit(ls(vs_layer_dir))';
nfiles=length(allfiles)-1;
for i=1:nfiles
    filename_prefix = allfiles{i}; 
    temp=load([vs_layer_dir    allfiles{i,1}]);
    vel_sws          = [temp(:,1:2),temp(:,3),ones(length(temp),1)*0.1]; profile_sws = vel_sws(:,1:2);
    vel_cnn_USA      = [temp(:,1:2),temp(:,4),ones(length(temp),1)*0.1]; profile_USA = vel_cnn_USA(:,1:2);
    vel_cnn_USATibet = [temp(:,1:2),temp(:,5),ones(length(temp),1)*0.1]; profile_USATibet = vel_cnn_USATibet(:,1:2);
     
    vs1 = krig_interp(profile_sws,vel_sws);
    vs2 = krig_interp(profile_USA,vel_cnn_USA);
    vs3 = krig_interp(profile_USATibet,vel_cnn_USATibet);
    temp(:,3) = vs1(:);
    temp(:,4) = vs2(:);
    temp(:,5) = vs3(:);
    save([vs_layer_dir    allfiles{i,1}],'temp','-ascii')
end
 


