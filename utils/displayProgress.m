function displayProgress(i, arrayLength)
% Function to display a progress update in the command window.
% Useful for "for" loops which take a while to complete.

persistent reverseStr tstart;
if i == 1
    reverseStr = '';
    tstart = tic; % start the timer
end

% Fillter to print
if ispc
    filler = char(9611);
else
    filler = char(9608);
end

% Print on command window
percentDone = 100 * i / arrayLength;
dt = toc(tstart);
iterPerSec = i / dt;
timeLeft   = (arrayLength - i) / iterPerSec;
barLength = 40;
barFilled = floor(barLength * i / arrayLength);
fills = [repmat(filler,1,barFilled), repmat(' ',1,barLength-barFilled)];
msg = sprintf('%3i%%%%|%s| %i/%i [%02i:%02i<%02i:%02i, %.2fit/s]', ...
    floor(percentDone), fills, ...
    i, arrayLength, ...
    floor(dt/60), floor(mod(dt,60)), ...
    floor(timeLeft/60), floor(mod(timeLeft,60)), ...
    iterPerSec);
fprintf([reverseStr, msg]);
reverseStr = repmat(sprintf('\b'), 1, length(msg)-1);

% Print extra line if at the end
if i == arrayLength
    fprintf(newline);
end
end
        