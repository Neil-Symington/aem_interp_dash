
dirList = glob('C:\Users\PCUser\Desktop\NSC_data\data\AEM\HE\garjmcmc\run4\output\pmaps\*.pmap');

% Loop through

for i = 1:length(dirList)

  fname = dirList{i,1};
  outfile = strrep(fname, '.pmap','.mat')                      
  read_pmap_file(fname, outfile);    
  

end