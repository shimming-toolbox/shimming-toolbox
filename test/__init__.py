#!/usr/bin/python3
#
# Original Author: Paul McCarthy <pauldmccarthy@gmail.com>
# Modified by: Alexandre D'Astous
#
# Originally from FSLeyes: https://git.fmrib.ox.ac.uk/fsl/fsleyes/fsleyes
# Modified for use by the Shimming Toolbox plugin
#

import os
import gc
import sys
import time
import traceback
import contextlib

import wx
from io import StringIO
from unittest import mock

import fsleyes_props as props
import fsl.utils.idle as idle
import fsleyes
import fsleyes.frame as fslframe
import fsleyes.main as fslmain
import fsleyes.actions.frameactions as frameactions  # noqa
import fsleyes.gl as fslgl
import fsleyes.colourmaps as colourmaps
import fsleyes.displaycontext as dc
import fsleyes.overlay as fsloverlay


def waitUntilIdle():

    called = [False]

    def flag():
        called[0] = True

    idle.idle(flag)

    while not called[0]:
        realYield(50)


@contextlib.contextmanager
def exitMainLoopOnError(app):

    oldhook = sys.excepthook

    error = [None]

    def myhook(type_, value, tb):

        # some errors come from
        # elsewhere (e.g. matplotlib),
        # and are out of our control
        ignore = True
        while tb is not None:
            frame = tb.tb_frame
            mod = frame.f_globals['__name__']

            if any([mod.startswith(m) for m in ('fsl', 'fsleyes')]):
                ignore = False
                break
            tb = tb.tb_next

        if not ignore:
            app.ExitMainLoop()
            error[0] = value

        oldhook(type_, value, traceback)

    try:
        sys.excepthook = myhook
        yield error
    finally:
        app = None
        sys.excepthook = oldhook


# Under GTK, a single call to
# yield just doesn't cut it
def realYield(centis=10):
    for i in range(int(centis)):
        wx.YieldIfNeeded()
        time.sleep(0.01)


def yieldUntil(condition):
    while not condition():
        realYield()


class CaptureStdout(object):
    """Context manager which captures stdout and stderr. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.__mock_stdout = StringIO('')
        self.__mock_stderr = StringIO('')

    def __enter__(self):
        self.__real_stdout = sys.stdout
        self.__real_stderr = sys.stderr

        sys.stdout = self.__mock_stdout
        sys.stderr = self.__mock_stderr

    def __exit__(self, *args, **kwargs):
        sys.stdout = self.__real_stdout
        sys.stderr = self.__real_stderr

        if args[0] is not None:
            print('Error')
            print('stdout:')
            print(self.stdout)
            print('stderr:')
            print(self.stderr)

        return False

    @property
    def stdout(self):
        self.__mock_stdout.seek(0)
        return self.__mock_stdout.read()

    @property
    def stderr(self):
        self.__mock_stderr.seek(0)
        return self.__mock_stderr.read()


def run_with_fsleyes(func, *args, **kwargs):
    """Create a ``FSLeyesFrame`` and run the given function. """

    import fsleyes_widgets.utils.status as status

    fsleyes.configLogging()

    gc.collect()
    idle.idleLoop.reset()
    idle.idleLoop.allowErrors = True

    propagateRaise = kwargs.pop('propagateRaise', True)
    startingDelay  = kwargs.pop('startingDelay',  500)
    finishingDelay = kwargs.pop('finishingDelay', 250)
    callAfterApp   = kwargs.pop('callAfterApp',   None)

    class State(object):
        pass
    state             = State()
    state.result      = None
    state.raised      = None
    state.frame       = None
    state.app         = None
    state.dummy       = None
    state.panel       = None

    glver  = os.environ.get('FSLEYES_TEST_GL', None)
    if glver is not None:
        glver = [int(v) for v in glver.split('.')]

    def init():
        fsleyes.initialise()
        props.initGUI()
        colourmaps.init()
        fslgl.bootstrap(glver)
        wx.CallAfter(run)

    def finish():
        state.frame.Close(askUnsaved=False, askLayout=False)
        state.dummy.Close()
        waitUntilIdle()
        realYield(100)
        fslgl.shutdown()
        state.app.ExitMainLoop()

    def run():

        overlayList = fsloverlay.OverlayList()
        displayCtx  = dc.DisplayContext(overlayList)
        state.frame = fslframe.FSLeyesFrame(None,
                                            overlayList,
                                            displayCtx)

        state.app.SetOverlayListAndDisplayContext(overlayList, displayCtx)
        state.app.SetTopWindow(state.frame)

        state.frame.Show()

        while not state.frame.IsShownOnScreen():
            realYield()

        try:
            if func is not None:
                state.result = func(state.frame,
                                    overlayList,
                                    displayCtx,
                                    *args,
                                    **kwargs)

        except Exception as e:
            traceback.print_exc()
            state.raised = e

        finally:
            wx.CallLater(finishingDelay, finish)

    state.app   = fslmain.FSLeyesApp()
    state.dummy = wx.Frame(None)
    state.panel = wx.Panel(state.dummy)
    state.sizer = wx.BoxSizer(wx.HORIZONTAL)
    state.sizer.Add(state.panel, flag=wx.EXPAND, proportion=1)
    state.dummy.SetSizer(state.sizer)

    if callAfterApp is not None:
        callAfterApp()

    state.dummy.SetSize((100, 100))
    state.dummy.Layout()
    state.dummy.Show()

    if getattr(fslgl, '_glContext', None) is not None:
        wx.CallLater(startingDelay, init)
    else:
        wx.CallLater(startingDelay,
                     fslgl.getGLContext,
                     ready=init,
                     raiseErrors=True,
                     requestVersion=glver)

    with exitMainLoopOnError(state.app) as err:
        state.app.MainLoop()

    status.setTarget(None)
    if status._clearThread is not None:
        status._clearThread.die()
        status._clearThread.clear(0.01)
        status._clearThread.join()
        status._clearThread = None

    raised = state.raised
    result = state.result

    if err[0] is not None:
        raise err[0]

    time.sleep(1)

    if raised and propagateRaise:
        raise raised

    state.app.Destroy()
    state = None

    return result


def run_with_viewpanel(func, vptype, *args, **kwargs):
    def inner(frame, overlayList, displayCtx, *a, **kwa):
        panel      = frame.addViewPanel(vptype)
        displayCtx = panel.displayCtx
        try:
            while not panel.IsShownOnScreen():
                realYield()
            result = func(panel, overlayList, displayCtx, *a, **kwa)
        except Exception as e:
            print(e)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise
        finally:
            frame.removeViewPanel(panel)
        return result
    return run_with_fsleyes(inner, *args, **kwargs)


def run_with_orthopanel(func, *args, **kwargs):
    """Create a ``FSLeyesFrame`` with an ``OrthoPanel`` and run the given
    function.
    """
    from fsleyes.views.orthopanel import OrthoPanel
    return run_with_viewpanel(func, OrthoPanel, *args, **kwargs)


@contextlib.contextmanager
def MockFileDialog(dirdlg=False):
    class MockDlg(object):
        def __init__(self, *args, **kwargs):
            pass
        def ShowModal(self):
            return MockDlg.ShowModal_retval
        def GetPath(self):
            return MockDlg.GetPath_retval
        def GetPaths(self):
            return MockDlg.GetPaths_retval
        def Close(self):
            pass
        def Destroy(self):
            pass
        ShowModal_retval = wx.ID_OK
        GetPath_retval   = ''
        GetPaths_retval  = []

    if dirdlg: patched = 'wx.DirDialog'
    else:      patched = 'wx.FileDialog'

    with mock.patch(patched, MockDlg):
        yield MockDlg


# stype:
#   0 for single click
#   1 for double click
#   2 for separatemouse down/up events
def simclick(sim, target, btn=wx.MOUSE_BTN_LEFT, pos=None, stype=0):

    GTK = any(['gtk' in p.lower() for p in wx.PlatformInfo])

    class FakeEv(object):
        def __init__(self, evo):
            self.evo = evo

        def GetEventObject(self):
            return self.evo

    parent = target.GetParent()
    if GTK:

        if type(target).__name__ == 'StaticTextTag' and \
           type(parent).__name__ == 'TextTagPanel':
            parent._TextTagPanel__onTagLeftDown(FakeEv(target))
            realYield()
            return

        if type(target).__name__ == 'StaticText' and \
           type(parent).__name__ == 'TogglePanel':
            parent.Toggle(FakeEv(target))
            realYield()
            return

    w, h = target.GetClientSize().Get()
    x, y = target.GetScreenPosition()

    if pos is None:
        pos = [0.5, 0.5]

    x += w * pos[0]
    y += h * pos[1]

    sim.MouseMove(round(x), round(y))
    realYield()
    if   stype == 0: sim.MouseClick(btn)
    elif stype == 1: sim.MouseDblClick(btn)
    else:
        sim.MouseDown(btn)
        sim.MouseUp(btn)
    realYield()


def simtext(sim, target, text, enter=True):

    GTK = any(['gtk' in p.lower() for p in wx.PlatformInfo])

    target.SetFocus()
    parent = target.GetParent()

    # The EVT_TEXT_ENTER event
    # does not seem to occur
    # under docker/GTK so we
    # have to hack. EVT_TEXT
    # does work though.
    if GTK and type(parent).__name__ == 'FloatSpinCtrl':
        if enter:
            target.ChangeValue(text)
            parent._FloatSpinCtrl__onText(None)
        else:
            target.SetValue(text)

    elif GTK and type(parent).__name__ == 'AutoTextCtrl':
        if enter:
            target.ChangeValue(text)
            parent._AutoTextCtrl__onEnter(None)
        else:
            target.SetValue(text)
    else:
        target.SetValue(text)

        if enter:
            sim.KeyDown(wx.WXK_RETURN)

    realYield()


def mockMouseEvent(profile, canvas, evType, canvasLoc):
    """Mock a mouse event on a SliceCanvas
    """
    # Uses intimate knowledge of the fsleyes.profiles.Profile class
    class MockEvent:
        def GetEventObject(self):
            return canvas
        def GetEventType(self):
            return {'LeftMouseDown' : wx.EVT_LEFT_DOWN.typeId,
                    'LeftMouseUp'   : wx.EVT_LEFT_UP.typeId,
                    'LeftMouseDrag' : wx.EVT_MOTION.typeId}[evType]
        def Dragging(self):
            return 'Drag' in evType
        def AltDown(self):
            return False
        def ControlDown(self):
            return False
        def ShiftDown(self):
            return False
        def Skip(self):
            pass
        def GetButton(self):
            if   'Left'   in evType: return wx.MOUSE_BTN_LEFT
            elif 'Right'  in evType: return wx.MOUSE_BTN_RIGHT
            elif 'Middle' in evType: return wx.MOUSE_BTN_MIDDLE
        def GetPosition(self):
            w, h = canvas.GetClientSize().Get()
            x, y = canvas.worldToCanvas(canvasLoc)
            return x, h - y

    profile.handleEvent(MockEvent())
