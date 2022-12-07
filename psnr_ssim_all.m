clc;close all;clear all;

NLD_value_ave = 0;
EPDN_value_ave = 0;
GDN_value_ave = 0;
DAN_value_ave = 0;
MSBDN_value_ave = 0;
KKKK_value_ave = 0;
PSD_value_ave = 0;
D4_value_ave = 0;
Former_value_ave = 0;
Dehamer_value_ave = 0;
ours_value_ave = 0;
input_value_ave = 0;

data_dir = 'D:\Dropbox\badweather\ACCV2022_defog\Dataset_day\Smoke_baselines\';
name_folder_dir =['D:\Dropbox\badweather\ACCV2022_defog\Dataset_day\Smoke\test\hazy\'];

img_name = dir([name_folder_dir, '*.png']);
num = length(img_name);

count = 0;
for i = 1:num
    disp(['Working on image: ' img_name(i).name]);
    [~,namepart, ext] = fileparts(img_name(i).name);

    name = namepart;
    % name = namepart(1:end-5);

    NLD_name = [data_dir 'berman/' namepart '.png']; 
    EPDN_name = [data_dir 'EPDN_19/' namepart '_final.png']; 
    GDN_name = [data_dir 'GDN_19/' namepart '.png']; 
    DAN_name = [data_dir 'DAN_20/' namepart '_r_dehazing_img.png']; 
    MSBDN_name = [data_dir 'MSBDN_20/' namepart '_MSBDN.png']; 
    KKKK_name = [data_dir '4K_21/' namepart '.png']; 
    PSD_name = [data_dir 'PSD_21/' namepart '.png'];
    D4_name = [data_dir 'D4_22/' namepart '_1.png'];
    Former_name = [data_dir 'DehazeFormer_22/' namepart '.png']; 
    Dehamer_name = [data_dir 'Dehamer_22/' namepart '.png']; 
    ours_name = ['D:\Dropbox\badweather\ACCV2022_defog\our\' namepart '.png']; 
    gt_name   = ['D:\Dropbox\badweather\ACCV2022_defog\Dataset_day\Smoke\test\clean\' name '.png']; 

    input_img = im2double(imread([name_folder_dir namepart '.png']));
    % input_img = im2double(imread([name_folder_dir namepart '.jpg']));
    [H W C] = size(input_img);

    NLD_img = im2double(imresize(imread(NLD_name),[H W]));
    EPDN_img = im2double(imresize(imread(EPDN_name),[H W]));
    GDN_img = im2double(imresize(imread(GDN_name),[H W]));
    DAN_img = im2double(imresize(imread(DAN_name),[H W]));
    MSBDN_img = im2double(imresize(imread(MSBDN_name),[H W]));
    KKKK_img = im2double(imresize(imread(KKKK_name),[H W]));
    PSD_img = im2double(imresize(imread(PSD_name),[H W]));
    D4_img = im2double(imresize(imread(D4_name),[H W]));
    Former_img = im2double(imresize(imread(Former_name),[H W]));
    Dehamer_img = im2double(imresize(imread(Dehamer_name),[H W]));
    ours_img = im2double(imresize(imread(ours_name),[H W]));
    gt_img = im2double(imresize(imread(gt_name),[H W]));
    
    NLD_value = psnr(NLD_img, gt_img);
    EPDN_value = psnr(EPDN_img, gt_img);
    GDN_value = psnr(GDN_img, gt_img);
    DAN_value = psnr(DAN_img, gt_img);
    MSBDN_value = psnr(MSBDN_img, gt_img);
    KKKK_value = psnr(KKKK_img, gt_img);
    PSD_value = psnr(PSD_img, gt_img);
    D4_value = psnr(D4_img, gt_img);
    Former_value = psnr(Former_img, gt_img);
    Dehamer_value = psnr(Dehamer_img, gt_img);
    ours_value = psnr(ours_img, gt_img);
    input_value = psnr(input_img, gt_img);

%     NLD_value = ssim(NLD_img, gt_img);
%     EPDN_value = ssim(EPDN_img, gt_img);
%     GDN_value = ssim(GDN_img, gt_img);
%     DAN_value = ssim(DAN_img, gt_img);
%     MSBDN_value = ssim(MSBDN_img, gt_img);
%     KKKK_value = ssim(KKKK_img, gt_img);
%     PSD_value = ssim(PSD_img, gt_img);
%     D4_value = ssim(D4_img, gt_img);
%     Former_value = ssim(Former_img, gt_img);
%     Dehamer_value = ssim(Dehamer_img, gt_img);
%     ours_value = ssim(ours_img, gt_img);
%     input_value = ssim(input_img, gt_img);

    NLD_value_ave = NLD_value_ave + NLD_value;
    EPDN_value_ave = EPDN_value_ave + EPDN_value;
    GDN_value_ave = GDN_value_ave + GDN_value;
    DAN_value_ave = DAN_value_ave + DAN_value;
    MSBDN_value_ave = MSBDN_value_ave + MSBDN_value;
    KKKK_value_ave = KKKK_value_ave + KKKK_value;
    PSD_value_ave = PSD_value_ave + PSD_value;
    D4_value_ave = D4_value_ave + D4_value;
    Former_value_ave = Former_value_ave + Former_value;
    Dehamer_value_ave = Dehamer_value_ave + Dehamer_value;
    ours_value_ave = ours_value_ave + ours_value;
    input_value_ave = input_value_ave + input_value;
    count = count + 1;
end

NLD_value_ave = NLD_value_ave/count
EPDN_value_ave = EPDN_value_ave/count
GDN_value_ave = GDN_value_ave/count
DAN_value_ave = DAN_value_ave/count
MSBDN_value_ave = MSBDN_value_ave/count
KKKK_value_ave = KKKK_value_ave/count
PSD_value_ave = PSD_value_ave/count
D4_value_ave = D4_value_ave/count
Former_value_ave = Former_value_ave/count
Dehamer_value_ave = Dehamer_value_ave/count
ours_value_ave = ours_value_ave/count
input_value_ave = input_value_ave/count
