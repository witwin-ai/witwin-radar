function compare_matlab_scenarios(directory)
% Actual Radar Toolbox APIs; synthetic formulas are labeled diagnostics.
identity.version = version;
identity.products = ver;
identity.radarTransceiver = which('radarTransceiver');
identity.reflectionModel = which('radar.scenario.SurfaceReflectionCoefficient');
assert(~isempty(identity.radarTransceiver) && ~isempty(identity.reflectionModel));
fid = fopen(fullfile(directory,'matlab-identity.json'),'w');
fprintf(fid,'%s',jsonencode(identity)); fclose(fid);
inputs = dir(fullfile(directory,'*-input.mat'));
assert(~isempty(inputs),'No exported experiment inputs found');
for index = 1:numel(inputs)
    name = erase(string(inputs(index).name),"-input.mat");
    data = load(fullfile(directory,inputs(index).name));
    if name == "materials"
        material_comparison(data,directory,name);
    elseif endsWith(name,"-performance")
        performance_comparison(data,directory,name);
    elseif endsWith(name,"-motion")
        if startsWith(name,"ground_")
            ground_comparison(data,directory,name);
        else
            motion_comparison(data,directory,name);
        end
    else
        error('WiTwin:UnknownComparison','Unknown comparison input: %s',name);
    end
end
end

function ground_comparison(data,directory,name)
factor = 1;
if contains(name,"_os4"), factor = 4; end
internal_sample_rate = data.fs*factor;
waveform = phased.FMCWWaveform('SampleRate',internal_sample_rate,'SweepTime',data.period, ...
    'SweepBandwidth',data.slope*data.period,'SweepInterval','Positive');
reference = waveform();
channel = twoRayChannel('OperatingFrequency',data.fc,'SampleRate',internal_sample_rate, ...
    'CombinedRaysOutput',false,'GroundReflectionCoefficient',1);
back_direct = clone(channel); back_reflected = clone(channel);
eps0 = 1/(4e-7*pi*299792458^2);
mdl = radar.scenario.SurfaceReflectionCoefficient(PermittivityModel=data.eps-1j*data.sigma/(2*pi*data.fc*eps0), ...
    HeightStandardDeviation=0,Slope=0,VegetationFactor="None");
matlab_iq_full = complex(zeros(double(data.samples)*factor,double(data.chirps)));
clock = tic;
for chirp = 1:double(data.chirps)
    t = (chirp-1)*data.period;
    x = double(single(30+0.3*t+15*t*t)); speed = double(single(0.3+30*t));
    tx = [0;0;30]; target = [x;0;30]; velocity = [speed;0;0];
    rho = reflectionCoefficient(mdl,data.fc,atan2d(60,x),'H');
    outgoing = channel(reference,tx,target,zeros(3,1),velocity);
    outgoing(:,2) = outgoing(:,2)*rho;
    outgoing = outgoing*sqrt(4*pi)/(299792458/data.fc); % unit RCS, same scalar scatter owner
    a = back_direct(outgoing(:,1),target,tx,velocity,zeros(3,1));
    b = back_reflected(outgoing(:,2),target,tx,velocity,zeros(3,1));
    received = a(:,1)+rho*a(:,2)+b(:,1)+rho*b(:,2);
    matlab_iq_full(:,chirp) = dechirp(received,reference);
end
seconds = toc(clock);
matlab_iq = matlab_iq_full(1:factor:end,:);
save(fullfile(directory,name+"-matlab.mat"),'matlab_iq','matlab_iq_full','internal_sample_rate','seconds','-v7');
end

function material_comparison(data,directory,name)
count = numel(data.eps);
matlab_coefficient = complex(zeros(1,count));
matlab_fresnel = complex(zeros(1,count));
eps0 = 1/(4e-7*pi*physconst('LightSpeed')^2);
for index = 1:count
    frequency = data.frequency(index);
    epsc = data.eps(index)-1j*data.sigma(index)/(2*pi*frequency*eps0);
    mdl = radar.scenario.SurfaceReflectionCoefficient(PermittivityModel=epsc, ...
        HeightStandardDeviation=0,Slope=0,VegetationFactor="None");
    rho = reflectionCoefficient(mdl,frequency,data.grazing(index),data.polarization(index));
    matlab_fresnel(index) = rho;
    % Independent finite-slab diagnostic: two air/dielectric boundaries.
    % Field convention exp(+j*w*t), normal thickness d in metres, mu_r=1.
    % The toolbox computes each interface; the Airy sum here is our oracle,
    % not a claim that Radar Toolbox natively traces this finite dielectric slab.
    kz = 2*pi*frequency/physconst('LightSpeed')*sqrt(epsc-cosd(data.grazing(index))^2);
    internal_roundtrip = exp(-2j*kz*data.thickness(index));
    matlab_coefficient(index) = rho*(1-internal_roundtrip)/(1-rho^2*internal_roundtrip);
end
save(fullfile(directory,name+"-matlab.mat"),'matlab_coefficient','matlab_fresnel','-v7');
end

function [radar,reference] = make_transceiver(data,repetitions)
waveform = phased.FMCWWaveform('SampleRate',data.fs,'SweepTime',data.period, ...
    'SweepBandwidth',data.slope*data.period,'SweepInterval','Positive');
antenna = phased.IsotropicAntennaElement('FrequencyRange',[1e9 100e9]);
radar = radarTransceiver('Waveform',waveform, ...
    'TransmitAntenna',phased.Radiator('Sensor',antenna,'OperatingFrequency',data.fc), ...
    'ReceiveAntenna',phased.Collector('Sensor',antenna,'OperatingFrequency',data.fc), ...
    'Transmitter',phased.Transmitter('PeakPower',1,'Gain',0), ...
    'Receiver',phased.ReceiverPreamp('Gain',0,'NoiseMethod','Noise power','NoisePower',0), ...
    'NumRepetitions',repetitions);
reference = waveform();
end

function performance_comparison(data,directory,name)
[radar,reference] = make_transceiver(data,double(data.chirps));
prototype = struct('PathLength',1,'PathLoss',0,'ReflectionCoefficient',1, ...
    'AngleOfDeparture',[0;0],'AngleOfArrival',[0;0],'DopplerShift',0);
paths = repmat(prototype,1,numel(data.lengths));
for index = 1:numel(paths)
    paths(index).PathLength = data.lengths(index);
    paths(index).ReflectionCoefficient = data.gains(index);
    paths(index).DopplerShift = -data.fc*data.rates(index)/299792458;
end
seconds = zeros(1,11);
for repetition = 1:14
    reset(radar);
    clock = tic;
    matlab_iq = dechirp(radar(paths,0),reference);
    elapsed = toc(clock);
    if repetition>3, seconds(repetition-3) = elapsed; end
end
output_class = class(matlab_iq);
save(fullfile(directory,name+"-matlab.mat"),'matlab_iq','seconds','output_class','-v7');
fprintf('%s median %.6f s\n',name,median(seconds));
end

function motion_comparison(data,directory,name)
base_name = erase(name,"_os4");
repetitions = 1;
if base_name == "static-motion", repetitions = double(data.chirps); end
[radar,reference] = make_transceiver(data,repetitions);
seconds = zeros(1,3);
for repetition = 1:4
    reset(radar);
    clock = tic;
    matlab_iq = scene_signal(radar,reference,data,base_name);
    elapsed = toc(clock);
    if repetition>1, seconds(repetition-1) = elapsed; end
end
save(fullfile(directory,name+"-matlab.mat"),'matlab_iq','seconds','-v7');
fprintf('%s median %.6f s\n',name,median(seconds));
end

function matlab_iq = scene_signal(radar,reference,data,name)
target = struct('Position',[0 0 0],'Velocity',[0 0 0]);
targets = repmat(target,1,size(data.positions,2));
if name == "static-motion"
    targets(1).Position = [30 0 0];
    matlab_iq = dechirp(radar(targets,0),reference);
    return
end
matlab_iq = complex(zeros(double(data.samples),double(data.chirps)));
for chirp = 1:double(data.chirps)
    t = (chirp-1)*data.period;
    w = 2*pi*80;
    if name == "acceleration-motion"
        p = [30+0.3*t+15*t*t,0,0]; v = [0.3+30*t,0,0];
    elseif name == "rotor-motion"
        p = [30+0.003*cos(w*t),0.003*sin(w*t),0];
        v = [-0.003*w*sin(w*t),0.003*w*cos(w*t),0];
    else
        p = [30+0.002*sin(w*t),0.1,0; 30.03+0.004*sin(w*t/2),-0.1,0];
        v = [0.002*w*cos(w*t),0,0; 0.002*w*cos(w*t/2),0,0];
    end
    % Match the authored float32 pose precision at the interface; waveform
    % computation remains native double in this Radar Toolbox object.
    p = double(single(p)); v = double(single(v));
    for index = 1:numel(targets)
        targets(index).Position = p(index,:);
        targets(index).Velocity = v(index,:);
    end
    matlab_iq(:,chirp) = dechirp(radar(targets,t),reference);
end
end
