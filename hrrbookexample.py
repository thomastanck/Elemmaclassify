from CircularHRR import *

n = 1024

cause, eat, see, hit = (CircularHRR(n) for _ in range(4))
being, human, state, food, fish, bread = (CircularHRR(n) for _ in range(6))
object_, agent = (CircularHRR(n) for _ in range(2))

idmark, idjohn, idpaul, idluke, idthefish, idthebread, idhunger, idthirst, ideatagent, ideatobject, idcauseagent, idcauseobject, idhitagent, idhitobject = (CircularHRR(n) for _ in range(14))

mark = (being + human + idmark).normalize()
john = (being + human + idjohn).normalize()
paul = (being + human + idpaul).normalize()
luke = (being + human + idluke).normalize()
thefish = (food + fish + idthefish).normalize()
thebread = (food + bread + idthebread).normalize()
hunger = (state + idhunger).normalize()
thirst = (state + idthirst).normalize()
eatagt = (agent + ideatagent).normalize()
eatobj = (object_ + ideatobject).normalize()
causeagt = (agent + idcauseagent).normalize()
causeobj = (object_ + idcauseobject).normalize()
hitagt = (agent + idhitagent).normalize()
hitobj = (object_ + idhitobject).normalize()

s1 = (eat + eatagt * mark + eatobj * thefish).normalize()
s2 = (cause + causeagt * hunger + causeobj * s1).normalize()

noncomms1 = (eat + eatagt @ mark + eatobj @ thefish).normalize()
noncomms2 = (cause + causeagt @ hunger + causeobj @ noncomms1).normalize()
noncomms3 = (hit + hitagt @ mark + hitobj @ john).normalize()
noncomms4 = (hit + hitagt @ paul + hitobj @ luke).normalize()
noncomms5 = (cause + causeagt @ noncomms3 + causeobj @ noncomms4).normalize()

mem = CircularHRR.cleanupmemory([
    cause, eat, see,
    being, human, state, food, fish, bread,
    object_, agent,

    idmark, idjohn, idpaul, idluke, idthefish, idthebread, idhunger, idthirst, ideatagent, ideatobject,

    mark,
    john,
    paul,
    luke,
    thefish,
    thebread,
    hunger,
    thirst,
    eatagt,
    eatobj,

    s1,
    s2,

    noncomms1,
    noncomms2,
    noncomms3,
    noncomms4,
    noncomms5,
    ])


