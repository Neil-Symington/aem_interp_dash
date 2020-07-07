function M=read_pmap_file(binfile, outfile)

%binfile = 'C:\Users\PCUser\Desktop\EK_data\AEM\garjmcmcmtdem\run2\output\pmaps\seq.00000071.313001.4942719.000000.pmap';
%outfile = 'C:\\Users\\PCUser\\Desktop\\EK_data\\AEM\\garjmcmcmtdem\\run2\\output\\hdf5_files\\seq.00000071.313001.4942719.000000.hdf5';
%open the files for reading
fp  = fopen(binfile,'rb');
M.binfile = binfile;
M.comment = fgets(fp);
v = sscanf(M.comment,'%f');
if(length(v)==7)    
    M.flightnumber=v(2);
    M.linenumber=v(3);
    M.fidnumber=v(4);
    M.easting=v(5);
    M.northing=v(6);
    M.ndata=v(7);    
end

M.starttime    = fgets(fp);
M.endtime      = fgets(fp);
str = fgets(fp); M.samplingtime = sscanf(str,'%f');

str = fgets(fp); M.nchain  = sscanf(str,'%d');
str = fgets(fp); M.nsample = sscanf(str,'%d');
str = fgets(fp); M.nburnin = sscanf(str,'%d');
str = fgets(fp); M.nnuisance = sscanf(str,'%d');

str = fgets(fp); M.nl_min  = sscanf(str,'%d');
str = fgets(fp); M.nl_max  = sscanf(str,'%d');

str = fgets(fp); M.p_param = sscanf(str(1:6),'%s');
str = fgets(fp); M.pmin = sscanf(str,'%f');
str = fgets(fp); M.pmax = sscanf(str,'%f');
str = fgets(fp); M.npcells = sscanf(str,'%d');

str = fgets(fp); M.v_param = sscanf(str(1:6),'%s');
str = fgets(fp); M.vmin = sscanf(str,'%f');
str = fgets(fp); M.vmax = sscanf(str,'%f');
str = fgets(fp); M.nvcells = sscanf(str,'%d');

str = fgets(fp); M.sd_v      = sscanf(str,'%f');
str = fgets(fp); M.sd_p      = sscanf(str,'%f');
str = fgets(fp); M.sd_bd_v   = sscanf(str,'%f');

str = fgets(fp); M.ar_birth  = sscanf(str,'%f');
str = fgets(fp); M.ar_death  = sscanf(str,'%f');
str = fgets(fp); M.ar_value  = sscanf(str,'%f');
str = fgets(fp); M.ar_move  = sscanf(str,'%f');
str = fgets(fp); M.ar_nuisance = sscanf(str,'%f');

%%Highest likelihood model
str = fgets(fp);
%%Lowest misfit
str = fgets(fp);

%% Nuisances
for i=1:1:M.nnuisance
    str = fgets(fp); 
    disp(str);
    M.nuisance(i)=readnuisance(str);
end
for i=1:1:M.nnuisance
    str = fgets(fp); 
    v=sscanf(str,'%f');
    M.ncov(i,:)=v;
end
for i=1:1:M.nnuisance
    str = fgets(fp); 
    v=sscanf(str,'%f');
    M.ncor(i,:)=v;
end

%%nuisances
for i=1:1:M.nnuisance        
    M.nuisance(i).value = fread(fp,n,'float32');    
end

%The binary int format was int32 prior to 2015

%%layer histogram
M.lhist.nbins  = M.nl_max-M.nl_min+1;
M.lhist.centre = M.nl_min:1:M.nl_max;
M.lhist.counts = fread(fp,M.lhist.nbins,'uint32');

%%changepoint histogram
M.cp=fread(fp,M.npcells,'uint32');

%%ppd 2d histogram
a=fread(fp,M.npcells*M.nvcells,'uint32');
M.f=reshape(a,M.nvcells,M.npcells)';
for ci=1:1:M.nchain    
    M.conv(ci).chain=fread(fp,1,'uint32');
    M.conv(ci).nsamples=fread(fp,1,'uint32'); 
    k = M.conv(ci).nsamples;      
    
    M.conv(ci).sample=fread(fp,k,'uint32');
    M.conv(ci).nlayers=fread(fp,k,'uint32');
    M.conv(ci).misfit=fread(fp,k,'float');
    M.conv(ci).logppd=fread(fp,k,'float');
    M.conv(ci).ar_valuechange=fread(fp,k,'float');
    M.conv(ci).ar_move=fread(fp,k,'float');
    M.conv(ci).ar_birth=fread(fp,k,'float');
    M.conv(ci).ar_death=fread(fp,k,'float');
    M.conv(ci).ar_nuisancechange=fread(fp,k,'float');
end

save ("-mat", outfile, "M")

fclose(fp);
end