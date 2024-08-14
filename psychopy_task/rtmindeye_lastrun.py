#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on Wed Aug 14 10:09:20 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from dynamic_feedback_string
run_wait_string = ""
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'rtmindeye'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '1',
    'session (1 2 or 3)': '3',
    'starting_run': '0',
    'skip_practice': '1',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['session (1 2 or 3)'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/ps6938/Documents/GitHub/real_time_mindEye2/psychopy_task/rtmindeye_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='prismaMonitor', color=[-0.0039, -0.0039, -0.0039], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.0039, -0.0039, -0.0039]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp_6') is None:
        # initialise key_resp_6
        key_resp_6 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_6',
        )
    if deviceManager.getDevice('press_if_repeat_2') is None:
        # initialise press_if_repeat_2
        press_if_repeat_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='press_if_repeat_2',
        )
    if deviceManager.getDevice('all_keys_pressed_2') is None:
        # initialise all_keys_pressed_2
        all_keys_pressed_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='all_keys_pressed_2',
        )
    if deviceManager.getDevice('key_resp_7') is None:
        # initialise key_resp_7
        key_resp_7 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_7',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('press_if_repeat') is None:
        # initialise press_if_repeat
        press_if_repeat = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='press_if_repeat',
        )
    if deviceManager.getDevice('all_keys_pressed') is None:
        # initialise all_keys_pressed
        all_keys_pressed = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='all_keys_pressed',
        )
    if deviceManager.getDevice('key_resp_5') is None:
        # initialise key_resp_5
        key_resp_5 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_5',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "set_up" ---
    
    # --- Initialize components for Routine "practice_continue" ---
    key_resp_6 = keyboard.Keyboard(deviceName='key_resp_6')
    text_6 = visual.TextStim(win=win, name='text_6',
        text='',
        font='Open Sans',
        units='deg', pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "practice_trial" ---
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='deg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(8.4, 8.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    press_if_repeat_2 = keyboard.Keyboard(deviceName='press_if_repeat_2')
    polygon_3 = visual.ShapeStim(
        win=win, name='polygon_3',units='deg', 
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=3.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    all_keys_pressed_2 = keyboard.Keyboard(deviceName='all_keys_pressed_2')
    
    # --- Initialize components for Routine "testing_continue" ---
    key_resp_7 = keyboard.Keyboard(deviceName='key_resp_7')
    text_7 = visual.TextStim(win=win, name='text_7',
        text='',
        font='Open Sans',
        units='deg', pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "waiting_fmri" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    text_3 = visual.TextStim(win=win, name='text_3',
        text='waiting for scanner…',
        font='Open Sans',
        units='deg', pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "set_skips_and_correct_key" ---
    # Run 'Begin Experiment' code from blank_trial_set_up
    not_stopping = False
    # Run 'Begin Experiment' code from set_correct_key
    correct_key_1or2 = '-1'
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='deg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(8.4, 8.4),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    press_if_repeat = keyboard.Keyboard(deviceName='press_if_repeat')
    polygon = visual.ShapeStim(
        win=win, name='polygon',units='deg', 
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=3.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.5, depth=-2.0, interpolate=True)
    # Run 'Begin Experiment' code from update_num_correct
    num_trials_correct = 0
    num_trials_total = 0
    num_trials_responded = 0
    all_keys_pressed = keyboard.Keyboard(deviceName='all_keys_pressed')
    
    # --- Initialize components for Routine "run_wait2_2" ---
    key_resp_5 = keyboard.Keyboard(deviceName='key_resp_5')
    # Run 'Begin Experiment' code from dynamic_feedback_string
    run_wait_string = ""
    # Run 'Begin Experiment' code from set_up_within_round_performance
    thisRun_num_trials_correct = 0
    thisRun_num_trials_total = 0
    thisRun_num_trials_responded = 0
    text_5 = visual.TextStim(win=win, name='text_5',
        text='',
        font='Open Sans',
        units='deg', pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "waiting_fmri_2" ---
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    text_4 = visual.TextStim(win=win, name='text_4',
        text='waiting for scanner…',
        font='Open Sans',
        units='deg', pos=(0, 0), height=0.5, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "set_up" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('set_up.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from code
    participant_conditions_file = "conditions_files/participant" + expInfo["participant"] + ".csv"
    practice_participant_conditions_file = "practice_conditions_files/practice.csv"
    
    starting_run = int(expInfo["starting_run"])
    run = starting_run
    print("starting_run: ", starting_run)
    
    session = int(expInfo["session (1 2 or 3)"])
    print("session (1 | 2 | 3): ", session)
    
    skip_practice = int(expInfo["skip_practice"])
    print("skip_practice: ", skip_practice)
    
    num_runs = 16 - starting_run
    print("num_runs: ", num_runs)
    
    first_cond_file = "conditions_files/participant" + expInfo["participant"] + "_run" + str(run) + "_sess" + str(session) + ".csv"
    print("starting condition file: ", first_cond_file)
    # keep track of which components have finished
    set_upComponents = []
    for thisComponent in set_upComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "set_up" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in set_upComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "set_up" ---
    for thisComponent in set_upComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('set_up.stopped', globalClock.getTime(format='float'))
    thisExp.nextEntry()
    # the Routine "set_up" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_continue" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('practice_continue.started', globalClock.getTime(format='float'))
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (skip_practice)
    # create starting attributes for key_resp_6
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    text_6.setText('Press any button to begin the practice trials.')
    # keep track of which components have finished
    practice_continueComponents = [key_resp_6, text_6]
    for thisComponent in practice_continueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_continue" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_6* updates
        waitOnFlip = False
        
        # if key_resp_6 is starting this frame...
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_6.started')
            # update status
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['1','2', '3','4'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_6.started')
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_continueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_continue" ---
    for thisComponent in practice_continueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('practice_continue.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    thisExp.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        thisExp.addData('key_resp_6.rt', key_resp_6.rt)
        thisExp.addData('key_resp_6.duration', key_resp_6.duration)
    thisExp.nextEntry()
    # the Routine "practice_continue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(practice_participant_conditions_file),
        seed=None, name='practice')
    thisExp.addLoop(practice)  # add the loop to the experiment
    thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
    if thisPractice != None:
        for paramName in thisPractice:
            globals()[paramName] = thisPractice[paramName]
    
    for thisPractice in practice:
        currentLoop = practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        # --- Prepare to start Routine "practice_trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('practice_trial.started', globalClock.getTime(format='float'))
        # skip this Routine if its 'Skip if' condition is True
        continueRoutine = continueRoutine and not (skip_practice)
        image_2.setImage(current_image)
        # create starting attributes for press_if_repeat_2
        press_if_repeat_2.keys = []
        press_if_repeat_2.rt = []
        _press_if_repeat_2_allKeys = []
        # create starting attributes for all_keys_pressed_2
        all_keys_pressed_2.keys = []
        all_keys_pressed_2.rt = []
        _all_keys_pressed_2_allKeys = []
        # keep track of which components have finished
        practice_trialComponents = [image_2, press_if_repeat_2, polygon_3, all_keys_pressed_2]
        for thisComponent in practice_trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_2* updates
            
            # if image_2 is starting this frame...
            if image_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_2.started')
                # update status
                image_2.status = STARTED
                image_2.setAutoDraw(True)
            
            # if image_2 is active this frame...
            if image_2.status == STARTED:
                # update params
                pass
            
            # if image_2 is stopping this frame...
            if image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_2.tStop = t  # not accounting for scr refresh
                    image_2.tStopRefresh = tThisFlipGlobal  # on global time
                    image_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.stopped')
                    # update status
                    image_2.status = FINISHED
                    image_2.setAutoDraw(False)
            
            # *press_if_repeat_2* updates
            waitOnFlip = False
            
            # if press_if_repeat_2 is starting this frame...
            if press_if_repeat_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                press_if_repeat_2.frameNStart = frameN  # exact frame index
                press_if_repeat_2.tStart = t  # local t and not account for scr refresh
                press_if_repeat_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_if_repeat_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_if_repeat_2.started')
                # update status
                press_if_repeat_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(press_if_repeat_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(press_if_repeat_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if press_if_repeat_2 is stopping this frame...
            if press_if_repeat_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > press_if_repeat_2.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    press_if_repeat_2.tStop = t  # not accounting for scr refresh
                    press_if_repeat_2.tStopRefresh = tThisFlipGlobal  # on global time
                    press_if_repeat_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'press_if_repeat_2.stopped')
                    # update status
                    press_if_repeat_2.status = FINISHED
                    press_if_repeat_2.status = FINISHED
            if press_if_repeat_2.status == STARTED and not waitOnFlip:
                theseKeys = press_if_repeat_2.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _press_if_repeat_2_allKeys.extend(theseKeys)
                if len(_press_if_repeat_2_allKeys):
                    press_if_repeat_2.keys = _press_if_repeat_2_allKeys[-1].name  # just the last key pressed
                    press_if_repeat_2.rt = _press_if_repeat_2_allKeys[-1].rt
                    press_if_repeat_2.duration = _press_if_repeat_2_allKeys[-1].duration
                    # was this correct?
                    if (press_if_repeat_2.keys == str(correct_key_1or2)) or (press_if_repeat_2.keys == correct_key_1or2):
                        press_if_repeat_2.corr = 1
                    else:
                        press_if_repeat_2.corr = 0
            
            # *polygon_3* updates
            
            # if polygon_3 is starting this frame...
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_3.started')
                # update status
                polygon_3.status = STARTED
                polygon_3.setAutoDraw(True)
            
            # if polygon_3 is active this frame...
            if polygon_3.status == STARTED:
                # update params
                pass
            
            # if polygon_3 is stopping this frame...
            if polygon_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_3.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_3.tStop = t  # not accounting for scr refresh
                    polygon_3.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_3.stopped')
                    # update status
                    polygon_3.status = FINISHED
                    polygon_3.setAutoDraw(False)
            
            # *all_keys_pressed_2* updates
            waitOnFlip = False
            
            # if all_keys_pressed_2 is starting this frame...
            if all_keys_pressed_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                all_keys_pressed_2.frameNStart = frameN  # exact frame index
                all_keys_pressed_2.tStart = t  # local t and not account for scr refresh
                all_keys_pressed_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(all_keys_pressed_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'all_keys_pressed_2.started')
                # update status
                all_keys_pressed_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(all_keys_pressed_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(all_keys_pressed_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if all_keys_pressed_2 is stopping this frame...
            if all_keys_pressed_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > all_keys_pressed_2.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    all_keys_pressed_2.tStop = t  # not accounting for scr refresh
                    all_keys_pressed_2.tStopRefresh = tThisFlipGlobal  # on global time
                    all_keys_pressed_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'all_keys_pressed_2.stopped')
                    # update status
                    all_keys_pressed_2.status = FINISHED
                    all_keys_pressed_2.status = FINISHED
            if all_keys_pressed_2.status == STARTED and not waitOnFlip:
                theseKeys = all_keys_pressed_2.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                _all_keys_pressed_2_allKeys.extend(theseKeys)
                if len(_all_keys_pressed_2_allKeys):
                    all_keys_pressed_2.keys = [key.name for key in _all_keys_pressed_2_allKeys]  # storing all keys
                    all_keys_pressed_2.rt = [key.rt for key in _all_keys_pressed_2_allKeys]
                    all_keys_pressed_2.duration = [key.duration for key in _all_keys_pressed_2_allKeys]
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice_trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_trial" ---
        for thisComponent in practice_trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('practice_trial.stopped', globalClock.getTime(format='float'))
        # check responses
        if press_if_repeat_2.keys in ['', [], None]:  # No response was made
            press_if_repeat_2.keys = None
            # was no response the correct answer?!
            if str(correct_key_1or2).lower() == 'none':
               press_if_repeat_2.corr = 1;  # correct non-response
            else:
               press_if_repeat_2.corr = 0;  # failed to respond (incorrectly)
        # store data for practice (TrialHandler)
        practice.addData('press_if_repeat_2.keys',press_if_repeat_2.keys)
        practice.addData('press_if_repeat_2.corr', press_if_repeat_2.corr)
        if press_if_repeat_2.keys != None:  # we had a response
            practice.addData('press_if_repeat_2.rt', press_if_repeat_2.rt)
            practice.addData('press_if_repeat_2.duration', press_if_repeat_2.duration)
        # check responses
        if all_keys_pressed_2.keys in ['', [], None]:  # No response was made
            all_keys_pressed_2.keys = None
        practice.addData('all_keys_pressed_2.keys',all_keys_pressed_2.keys)
        if all_keys_pressed_2.keys != None:  # we had a response
            practice.addData('all_keys_pressed_2.rt', all_keys_pressed_2.rt)
            practice.addData('all_keys_pressed_2.duration', all_keys_pressed_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'practice'
    
    
    # --- Prepare to start Routine "testing_continue" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('testing_continue.started', globalClock.getTime(format='float'))
    # skip this Routine if its 'Skip if' condition is True
    continueRoutine = continueRoutine and not (skip_practice)
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    text_7.setText('Good job! You finished the practice trials. Press any button to begin the experiment.')
    # keep track of which components have finished
    testing_continueComponents = [key_resp_7, text_7]
    for thisComponent in testing_continueComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "testing_continue" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in testing_continueComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "testing_continue" ---
    for thisComponent in testing_continueComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('testing_continue.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "testing_continue" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "waiting_fmri" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('waiting_fmri.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    waiting_fmriComponents = [key_resp_2, text_3]
    for thisComponent in waiting_fmriComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "waiting_fmri" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['=', 'equal'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in waiting_fmriComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "waiting_fmri" ---
    for thisComponent in waiting_fmriComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('waiting_fmri.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "waiting_fmri" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_runs = data.TrialHandler(nReps=num_runs, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_runs')
    thisExp.addLoop(loop_runs)  # add the loop to the experiment
    thisLoop_run = loop_runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_run.rgb)
    if thisLoop_run != None:
        for paramName in thisLoop_run:
            globals()[paramName] = thisLoop_run[paramName]
    
    for thisLoop_run in loop_runs:
        currentLoop = loop_runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_run.rgb)
        if thisLoop_run != None:
            for paramName in thisLoop_run:
                globals()[paramName] = thisLoop_run[paramName]
        
        # set up handler to look after randomisation of conditions etc
        loop_all_images = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions("conditions_files/participant" + expInfo["participant"] + "_run" + str(run) + "_sess" + str(session) + ".csv"),
            seed=None, name='loop_all_images')
        thisExp.addLoop(loop_all_images)  # add the loop to the experiment
        thisLoop_all_image = loop_all_images.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_all_image.rgb)
        if thisLoop_all_image != None:
            for paramName in thisLoop_all_image:
                globals()[paramName] = thisLoop_all_image[paramName]
        
        for thisLoop_all_image in loop_all_images:
            currentLoop = loop_all_images
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_all_image.rgb)
            if thisLoop_all_image != None:
                for paramName in thisLoop_all_image:
                    globals()[paramName] = thisLoop_all_image[paramName]
            
            # --- Prepare to start Routine "set_skips_and_correct_key" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('set_skips_and_correct_key.started', globalClock.getTime(format='float'))
            # Run 'Begin Routine' code from blank_trial_set_up
            if is_new_run:
                not_stopping = False
            else:
                not_stopping = True
            # Run 'Begin Routine' code from set_correct_key
            if is_repeat:
                correct_key_1or2 = '2'
            else:
                correct_key_1or2 = '1'
            
            # keep track of which components have finished
            set_skips_and_correct_keyComponents = []
            for thisComponent in set_skips_and_correct_keyComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "set_skips_and_correct_key" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in set_skips_and_correct_keyComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "set_skips_and_correct_key" ---
            for thisComponent in set_skips_and_correct_keyComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('set_skips_and_correct_key.stopped', globalClock.getTime(format='float'))
            # the Routine "set_skips_and_correct_key" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime(format='float'))
            image.setImage(current_image)
            # create starting attributes for press_if_repeat
            press_if_repeat.keys = []
            press_if_repeat.rt = []
            _press_if_repeat_allKeys = []
            # create starting attributes for all_keys_pressed
            all_keys_pressed.keys = []
            all_keys_pressed.rt = []
            _all_keys_pressed_allKeys = []
            # keep track of which components have finished
            trialComponents = [image, press_if_repeat, polygon, all_keys_pressed]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *press_if_repeat* updates
                waitOnFlip = False
                
                # if press_if_repeat is starting this frame...
                if press_if_repeat.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    press_if_repeat.frameNStart = frameN  # exact frame index
                    press_if_repeat.tStart = t  # local t and not account for scr refresh
                    press_if_repeat.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(press_if_repeat, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'press_if_repeat.started')
                    # update status
                    press_if_repeat.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(press_if_repeat.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(press_if_repeat.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if press_if_repeat is stopping this frame...
                if press_if_repeat.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > press_if_repeat.tStartRefresh + 3-frameTolerance:
                        # keep track of stop time/frame for later
                        press_if_repeat.tStop = t  # not accounting for scr refresh
                        press_if_repeat.tStopRefresh = tThisFlipGlobal  # on global time
                        press_if_repeat.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'press_if_repeat.stopped')
                        # update status
                        press_if_repeat.status = FINISHED
                        press_if_repeat.status = FINISHED
                if press_if_repeat.status == STARTED and not waitOnFlip:
                    theseKeys = press_if_repeat.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _press_if_repeat_allKeys.extend(theseKeys)
                    if len(_press_if_repeat_allKeys):
                        press_if_repeat.keys = _press_if_repeat_allKeys[-1].name  # just the last key pressed
                        press_if_repeat.rt = _press_if_repeat_allKeys[-1].rt
                        press_if_repeat.duration = _press_if_repeat_allKeys[-1].duration
                        # was this correct?
                        if (press_if_repeat.keys == str(correct_key_1or2)) or (press_if_repeat.keys == correct_key_1or2):
                            press_if_repeat.corr = 1
                        else:
                            press_if_repeat.corr = 0
                
                # *polygon* updates
                
                # if polygon is starting this frame...
                if polygon.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    polygon.frameNStart = frameN  # exact frame index
                    polygon.tStart = t  # local t and not account for scr refresh
                    polygon.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.started')
                    # update status
                    polygon.status = STARTED
                    polygon.setAutoDraw(True)
                
                # if polygon is active this frame...
                if polygon.status == STARTED:
                    # update params
                    pass
                
                # if polygon is stopping this frame...
                if polygon.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > polygon.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        polygon.tStop = t  # not accounting for scr refresh
                        polygon.tStopRefresh = tThisFlipGlobal  # on global time
                        polygon.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'polygon.stopped')
                        # update status
                        polygon.status = FINISHED
                        polygon.setAutoDraw(False)
                
                # *all_keys_pressed* updates
                waitOnFlip = False
                
                # if all_keys_pressed is starting this frame...
                if all_keys_pressed.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    all_keys_pressed.frameNStart = frameN  # exact frame index
                    all_keys_pressed.tStart = t  # local t and not account for scr refresh
                    all_keys_pressed.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(all_keys_pressed, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'all_keys_pressed.started')
                    # update status
                    all_keys_pressed.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(all_keys_pressed.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(all_keys_pressed.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if all_keys_pressed is stopping this frame...
                if all_keys_pressed.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > all_keys_pressed.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        all_keys_pressed.tStop = t  # not accounting for scr refresh
                        all_keys_pressed.tStopRefresh = tThisFlipGlobal  # on global time
                        all_keys_pressed.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'all_keys_pressed.stopped')
                        # update status
                        all_keys_pressed.status = FINISHED
                        all_keys_pressed.status = FINISHED
                if all_keys_pressed.status == STARTED and not waitOnFlip:
                    theseKeys = all_keys_pressed.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                    _all_keys_pressed_allKeys.extend(theseKeys)
                    if len(_all_keys_pressed_allKeys):
                        all_keys_pressed.keys = [key.name for key in _all_keys_pressed_allKeys]  # storing all keys
                        all_keys_pressed.rt = [key.rt for key in _all_keys_pressed_allKeys]
                        all_keys_pressed.duration = [key.duration for key in _all_keys_pressed_allKeys]
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
            # check responses
            if press_if_repeat.keys in ['', [], None]:  # No response was made
                press_if_repeat.keys = None
                # was no response the correct answer?!
                if str(correct_key_1or2).lower() == 'none':
                   press_if_repeat.corr = 1;  # correct non-response
                else:
                   press_if_repeat.corr = 0;  # failed to respond (incorrectly)
            # store data for loop_all_images (TrialHandler)
            loop_all_images.addData('press_if_repeat.keys',press_if_repeat.keys)
            loop_all_images.addData('press_if_repeat.corr', press_if_repeat.corr)
            if press_if_repeat.keys != None:  # we had a response
                loop_all_images.addData('press_if_repeat.rt', press_if_repeat.rt)
                loop_all_images.addData('press_if_repeat.duration', press_if_repeat.duration)
            # Run 'End Routine' code from update_num_correct
            if not is_blank_trial:
                thisRun_num_trials_total += 1 
                if press_if_repeat.corr == 1:
                    thisRun_num_trials_correct += 1
                if press_if_repeat.keys != None:
                    thisRun_num_trials_responded += 1
            # check responses
            if all_keys_pressed.keys in ['', [], None]:  # No response was made
                all_keys_pressed.keys = None
            loop_all_images.addData('all_keys_pressed.keys',all_keys_pressed.keys)
            if all_keys_pressed.keys != None:  # we had a response
                loop_all_images.addData('all_keys_pressed.rt', all_keys_pressed.rt)
                loop_all_images.addData('all_keys_pressed.duration', all_keys_pressed.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "run_wait2_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('run_wait2_2.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (not_stopping)
            # create starting attributes for key_resp_5
            key_resp_5.keys = []
            key_resp_5.rt = []
            _key_resp_5_allKeys = []
            # Run 'Begin Routine' code from dynamic_feedback_string
            insert_num_str = (thisRun_num_trials_correct / thisRun_num_trials_total)
            if is_new_run:
                if run_num < 15:
                    run += 1
                    run_wait_string = "Good job! Your average accuracy in this run is " + f"{insert_num_str:.3f}" + \
                                    ". Please press any button when you are ready to continue to run " + str(run_num + 2) + "/16" 
                else: 
                    run_wait_string = "Good job! Your average accuracy in this run is " + f"{insert_num_str:.3f}" + \
                    ". You completed all the runs!" 
            text_5.setText(run_wait_string
            )
            # keep track of which components have finished
            run_wait2_2Components = [key_resp_5, text_5]
            for thisComponent in run_wait2_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "run_wait2_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *key_resp_5* updates
                waitOnFlip = False
                
                # if key_resp_5 is starting this frame...
                if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_5.frameNStart = frameN  # exact frame index
                    key_resp_5.tStart = t  # local t and not account for scr refresh
                    key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_5.started')
                    # update status
                    key_resp_5.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_5.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_5.getKeys(keyList=['1','2','3','4'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_5_allKeys.extend(theseKeys)
                    if len(_key_resp_5_allKeys):
                        key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                        key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                        key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_5* updates
                
                # if text_5 is starting this frame...
                if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_5.frameNStart = frameN  # exact frame index
                    text_5.tStart = t  # local t and not account for scr refresh
                    text_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_5.started')
                    # update status
                    text_5.status = STARTED
                    text_5.setAutoDraw(True)
                
                # if text_5 is active this frame...
                if text_5.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in run_wait2_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "run_wait2_2" ---
            for thisComponent in run_wait2_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('run_wait2_2.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp_5.keys in ['', [], None]:  # No response was made
                key_resp_5.keys = None
            loop_all_images.addData('key_resp_5.keys',key_resp_5.keys)
            if key_resp_5.keys != None:  # we had a response
                loop_all_images.addData('key_resp_5.rt', key_resp_5.rt)
                loop_all_images.addData('key_resp_5.duration', key_resp_5.duration)
            # Run 'End Routine' code from set_up_within_round_performance
            if is_new_run:
                thisRun_num_trials_correct = 0
                thisRun_num_trials_total = 0
                thisRun_num_trials_responded = 0
            # the Routine "run_wait2_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "waiting_fmri_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('waiting_fmri_2.started', globalClock.getTime(format='float'))
            # skip this Routine if its 'Skip if' condition is True
            continueRoutine = continueRoutine and not (not_stopping)
            # create starting attributes for key_resp_4
            key_resp_4.keys = []
            key_resp_4.rt = []
            _key_resp_4_allKeys = []
            # keep track of which components have finished
            waiting_fmri_2Components = [key_resp_4, text_4]
            for thisComponent in waiting_fmri_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "waiting_fmri_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *key_resp_4* updates
                waitOnFlip = False
                
                # if key_resp_4 is starting this frame...
                if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_4.frameNStart = frameN  # exact frame index
                    key_resp_4.tStart = t  # local t and not account for scr refresh
                    key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_4.started')
                    # update status
                    key_resp_4.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_4.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_4.getKeys(keyList=['=', 'equal'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_4_allKeys.extend(theseKeys)
                    if len(_key_resp_4_allKeys):
                        key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                        key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                        key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_4* updates
                
                # if text_4 is starting this frame...
                if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_4.frameNStart = frameN  # exact frame index
                    text_4.tStart = t  # local t and not account for scr refresh
                    text_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.started')
                    # update status
                    text_4.status = STARTED
                    text_4.setAutoDraw(True)
                
                # if text_4 is active this frame...
                if text_4.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in waiting_fmri_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "waiting_fmri_2" ---
            for thisComponent in waiting_fmri_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('waiting_fmri_2.stopped', globalClock.getTime(format='float'))
            # check responses
            if key_resp_4.keys in ['', [], None]:  # No response was made
                key_resp_4.keys = None
            loop_all_images.addData('key_resp_4.keys',key_resp_4.keys)
            if key_resp_4.keys != None:  # we had a response
                loop_all_images.addData('key_resp_4.rt', key_resp_4.rt)
                loop_all_images.addData('key_resp_4.duration', key_resp_4.duration)
            # the Routine "waiting_fmri_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'loop_all_images'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed num_runs repeats of 'loop_runs'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
