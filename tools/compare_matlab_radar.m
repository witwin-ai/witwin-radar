function compare_matlab_radar(directory)
% Actual Radar Toolbox half. No fallback to an analytic or Python oracle.
% Run: addpath('tools'); compare_matlab_radar('output/doppler-repair/matlab')
% Scope: specified round-trip paths and ideal hardware, not scene discovery.
identity.version = version;
identity.products = ver;
identity.radarTransceiver = which('radarTransceiver');
identity.radarLicense = license('test','Radar_Toolbox');
identity.phasedLicense = license('test','Phased_Array_System_Toolbox');
fid = fopen(fullfile(directory,'matlab-identity.json'),'w');
assert(fid >= 0, 'Cannot write MATLAB identity');
fprintf(fid,'%s',jsonencode(identity)); fclose(fid);
assert(~isempty(identity.radarTransceiver) && identity.radarLicense, ...
    'WiTwin:MissingRadarToolbox','Radar Toolbox radarTransceiver is required');
inputs = dir(fullfile(directory,'*-input.mat'));
assert(~isempty(inputs), 'No exported WiTwin inputs found');
for inputIndex = 1:numel(inputs)
    name = erase(string(inputs(inputIndex).name),"-input.mat");
    data = load(fullfile(directory,name+"-input.mat"));
    waveform = phased.FMCWWaveform('SampleRate',data.fs, ...
        'SweepTime',data.period,'SweepBandwidth',data.slope*data.period, ...
        'SweepInterval','Positive');
    antenna = phased.IsotropicAntennaElement('FrequencyRange',[1e9 100e9]);
    radar = radarTransceiver('Waveform',waveform, ...
        'TransmitAntenna',phased.Radiator('Sensor',antenna,'OperatingFrequency',data.fc), ...
        'ReceiveAntenna',phased.Collector('Sensor',antenna,'OperatingFrequency',data.fc), ...
        'Transmitter',phased.Transmitter('PeakPower',1,'Gain',0), ...
        'Receiver',phased.ReceiverPreamp('Gain',0, ...
            'NoiseMethod','Noise power','NoisePower',0), ...
        'NumRepetitions',1);
    reference = waveform();
    prototype = struct('PathLength',1,'PathLoss',0,'ReflectionCoefficient',1, ...
        'AngleOfDeparture',[0;0],'AngleOfArrival',[0;0],'DopplerShift',0);
    paths = repmat(prototype,1,numel(data.lengths));
    matlab_iq = complex(zeros(data.samples,data.chirps));
    tic;
    for chirp = 1:double(data.chirps)
        t = (chirp-1)*data.period;
        for path = 1:numel(paths)
            paths(path).PathLength = data.lengths(path)+data.rates(path)*t;
            paths(path).ReflectionCoefficient = data.gains(path);
            paths(path).DopplerShift = -data.fc*data.rates(path)/299792458;
        end
        received = radar(paths,t);
        matlab_iq(:,chirp) = dechirp(received,reference);
    end
    matlab_seconds = toc;
    save(fullfile(directory,name+"-matlab.mat"),'matlab_iq','matlab_seconds','identity','-v7');
end
end
