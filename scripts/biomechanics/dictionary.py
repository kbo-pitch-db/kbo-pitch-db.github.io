"""Research concepts, operational adaptations and blocked measurements are distinct."""
VERSION='biomechanics-db-20261029-v2'
OBP='https://github.com/drivelineresearch/openbiomechanics/blob/main/baseball_pitching/README.md'
SOURCES={
 'obp':dict(url=OBP,title='OpenBiomechanics pitching README, POI dictionary and Visual3D pipeline',access='documentation_code_and_public_signals',doi=None,notes='Force contact 10% BW and plant 100% BW differ from image contact. Lab +X plate, +Y first base, +Z up. Visual3D segment definitions are not SMPL joint-center definitions.'),
 'report':dict(url='https://drivelinebaseball.com/blogs/blog/interpret-biomechanics-reports',title='How to interpret biomechanics reports',access='full_web_article',doi=None,notes='Field priorities: positions, velocities, sequencing and kinetics. Internal reference distributions are not KBO thresholds.'),
 'block':dict(url='https://drivelinebaseball.com/blogs/blog/a-quantitative-analysis-of-the-lead-leg-block-and-its-contributions-to-velocity',title='A quantitative analysis of the lead-leg block',access='full_web_article',doi=None,notes='Knee extension, additional flexion and ground-force components are distinct; knee motion does not measure braking force.'),
 'knee':dict(url='https://pubmed.ncbi.nlm.nih.gov/39157018/',doi='10.1177/23259671241257539',title='Lead-knee extension and pitching velocity',access='full_text_PMC11329978',notes='480 Hz, 13.4 Hz fourth-order zero-lag filter; image/marker first toe/heel contact, BR 0.01 s after wrist passes elbow in +X. Our release is an image bracket, not that model event.'),
 'sequence':dict(url='https://pubmed.ncbi.nlm.nih.gov/34896252/',doi='10.1016/j.jse.2021.10.045',title='Professional-pitcher segment sequencing',access='abstract_only',notes='287 professional pitchers; peak order and intervals are kinematic observations, not a measured energy-transfer score. Full operational methods not accessible; do not claim exact reproduction.'),
 'pelvis':dict(url='https://pubmed.ncbi.nlm.nih.gov/36479467/',doi='10.1177/23259671221130340',title='Pelvic rotation in professional pitchers',access='full_text_PMC9720809',notes='ASIS marker line, closed 90/open 0 convention and PKH rear-ankle to FC lead-ankle stride differ from the chosen rest-calibrated SMPL frame and same-frame ankle stride.'),
 'game':dict(url='https://pubmed.ncbi.nlm.nih.gov/41389727/',doi='10.1016/j.jbiomech.2025.113117',title='In-game hip–shoulder separation',access='abstract_only',notes='335 NCAA pitchers; weak time-local association and individual variation. Full methods inaccessible; motivates waveform/interval analysis, not a normative range or causal conclusion.'),
 'review':dict(url='https://pubmed.ncbi.nlm.nih.gov/38035212/',doi='10.1177/23259671231196539',title='Systematic review of kinematics associated with pitching velocity',access='full_text_PMC10687953',notes='Heterogeneous definitions across 12 studies / 930 participants. Associations do not establish cause or universal optimal mechanics.'),
 'accuracy':dict(url='https://pubmed.ncbi.nlm.nih.gov/41294254/',doi='10.1080/02640414.2025.2595411',title='Metric-specific markerless pitching accuracy',access='abstract_only',notes='18 pitchers, multi-camera Theia/Hawk-Eye comparison to markers. Variable-specific agreement does not validate this monocular broadcast pipeline.')}


def dictionary():
    metrics=[]
    def add(id,ko,en,group,series=None,event='fc',operation='event',unit='°',formula='',refs=('obp',),inputs=None,difference='',correspondence='operational_adaptation',blocked=None,sign=None):
        interval='image FC → image BR' if event=='fc_br' else ('PKH → BR' if event=='pkh_br' else 'image '+event.upper())
        definition=formula or en
        metrics.append(dict(id=id,metric_id=id,name={'ko':ko,'en':en},group=group,basis='3d',unit=unit,
          definition={'ko':definition,'en':definition},analysisInterval=interval,referenceEvent=event,
          coordinateSystem='Clip-fixed inferred up; X plate direction only where source-reviewed; Y=Z×X. SMPL rest-calibrated pelvis (root), thorax (spine3 cumulative). Intrinsic ZYX RzRyRx. Lean fields use two signed global-plane inclinations, not Cardan pitch/roll; see each formula. No anatomical marker calibration.',
          sign=sign or 'Flexion 0° straight; positive flexion. Signed rotations normalized for throwing side; see formula.',
          formula=definition,implementation='scripts/biomechanics/geometry.py; scripts/biomechanics/calculation.py',
          requiredJoints=inputs or series or blocked or id,requiredInputs=inputs or ['SMPL24 native joint centers and local rotations','image event brackets','original joint confidence'],
          sources=[dict(key=k,**SOURCES[k]) for k in refs],sourceUrls=[SOURCES[k]['url'] for k in refs],
          differenceFromSource=difference or 'Image event brackets replace force-based FC/FP. Native SMPL joint centers/rest axes replace laboratory landmarks; definitions are related but not numerically interchangeable.',
          definitionCorrespondence=correspondence,calculationVersion=VERSION,
          conditions='Known throwing role; supported original images; finite 3D; event interval present. Candidate automatic events remain diagnostic, not accepted observations. Physical velocities/times require a documented physical clock.',
          limitations='Monocular model estimate; no external biomechanical ground truth. Stability is not accuracy; do not apply laboratory normal ranges.',
          externalAccuracyValidation='not_validated_against_external_GT',series=series,operation=operation,blockedReason=blocked))
    add('lead_knee_flexion_fc','착지 시 앞무릎 굴곡','Lead-knee flexion at image contact','lead_leg','lead_knee_flexion',formula='3D hip–knee–ankle flexion = 180° − included angle; evaluate at image FC.',refs=('knee','block','obp'))
    add('lead_knee_flexion_br','릴리스 시 앞무릎 굴곡','Lead-knee flexion at release','lead_leg','lead_knee_flexion','br',formula='180° − 3D hip–knee–ankle included angle at image ball release.',refs=('knee','obp'))
    add('lead_knee_net_extension','착지→릴리스 순신전량','Net lead-knee extension FC→BR','lead_leg','lead_knee_flexion','fc_br','net_extension',formula='Flexion(FC) − flexion(BR); positive = net extension, negative = net additional flexion.',refs=('knee','block','obp'))
    add('lead_knee_max_flexion','착지→릴리스 최대 앞무릎 굴곡','Maximum lead-knee flexion FC→BR','lead_leg','lead_knee_flexion','fc_br','maximum',formula='max flexion among supported independent source images from FC through BR.',refs=('knee','block'))
    add('lead_knee_extra_flexion','착지 이후 추가 굴곡량','Additional lead-knee flexion after FC','lead_leg','lead_knee_flexion','fc_br','extra_flexion',formula='max flexion(FC…BR) − flexion(FC); nonnegative.',refs=('block','knee'))
    add('lead_knee_extension_transition_pct','최대 굴곡 이후 신전 전환 시점','Post-maximum-flexion extension transition','lead_leg','lead_knee_flexion','fc_br','transition',unit='% interval',formula='PROJECT DERIVED: first sustained decrease after maximum knee flexion; 2 subsequent independent observed images must decrease by ≥1° total; 100×(source timestamp − FC)/(BR − FC). Playback-phase only; physical linearity is unverified.',refs=('block',),correspondence='project_derived',difference='Not a named standardized biomechanical variable; source timestamps give playback phase, not a certified physical phase.')
    for ser,ko,en in [('pelvis_rotation','골반 회전','Pelvis rotation'),('trunk_rotation','몸통 축회전','Trunk axial rotation'),('trunk_forward_lean','몸통 전경','Trunk forward lean'),('trunk_lateral_lean','몸통 측굴','Trunk lateral lean')]:
        for event in ['fc','br'] if ser!='pelvis_rotation' else ['fc']:
            add(ser+'_'+event,ko+(' · 착지' if event=='fc' else ' · 릴리스'),en+' at '+event.upper(),'posture' if 'lean' in ser else 'pelvis_trunk',ser,event,
                formula={'pelvis_rotation':'Throwing-side-normalized yaw of rest-calibrated SMPL pelvis in pitch frame; 0° anterior faces plate; −90° closed (opposite sign to the cited 90° closed convention).','trunk_rotation':'Throwing-side-normalized intrinsic ZYX yaw of rest-calibrated SMPL spine3 thorax in pitch frame.','trunk_forward_lean':'atan2(thorax superior · plate X, thorax superior · up Z). Signed global sagittal-plane inclination; positive toward plate. This is a planar inclination adaptation, not Visual3D Cardan pitch.','trunk_lateral_lean':'hand_sign × atan2(thorax superior · pitch Y, thorax superior · up Z). Signed global frontal-plane inclination, positive glove-side; separate from forward tilt and axial rotation. Not Visual3D Cardan roll.'}[ser],refs=('obp','pelvis','review'))
        if ser!='pelvis_rotation':
            add(ser+'_change',ko+' 변화량 · 착지→릴리스',en+' change FC→BR','posture',ser,'fc_br','change',formula='Signed event endpoint difference BR − FC after within-interval angle unwrapping; intrinsic ZYX, not a vertical-line total tilt.',refs=('obp','review'),correspondence='project_derived')
    add('thorax_pelvis_axial_fc','착지 시 흉곽–골반 상대 축회전','Relative thorax–pelvis axial angle at FC','pelvis_trunk','thorax_pelvis_relative_axial',formula='−hand_sign × yaw_ZYX(Q_pelvisᵀ Q_thorax); model-rest segment frames. Positive = pelvis leads thorax. NOT shoulder-line separation.',refs=('obp','pelvis'),sign='R +1 / L −1; positive pelvis-leading; range [−180°,180°).')
    add('thorax_pelvis_axial_max','최대 흉곽–골반 상대 축회전','Maximum relative thorax–pelvis axial angle','pelvis_trunk','thorax_pelvis_relative_axial','fc_br','maximum',formula='Maximum signed pelvis-leading relative axial angle within FC→BR; unwrap locally; not absolute magnitude.',refs=('obp','game'))
    add('thorax_pelvis_axial_peak_pct','최대 흉곽–골반 상대 축회전 시점','Peak relative axial angle phase','pelvis_trunk','thorax_pelvis_relative_axial','fc_br','argmax',unit='% interval',formula='PROJECT DERIVED: 100×playback timestamp of maximum signed relative axial angle within FC→BR. Not physical milliseconds.',refs=('obp','game'),correspondence='project_derived')
    add('hip_shoulder_line_separation_fc','착지 시 골반선–어깨선 분리각','Hip–shoulder line separation at FC','pelvis_trunk','shoulder_line_separation',formula='−hand_sign × signed angle from R−L hip-center line to R−L shoulder-center line projected onto inferred horizontal plane. atan2(up·(a×b),a·b). Positive pelvis-leading. NOT thorax–pelvis Cardan rotation.',refs=('game','review'),difference='Joint-center horizontal line angle; full cited in-game methods unavailable. Shoulder/collar motion contributes; kept separate from thorax–pelvis field.')
    add('hip_shoulder_line_separation_max','최대 골반선–어깨선 분리각','Maximum hip–shoulder line separation','pelvis_trunk','shoulder_line_separation','fc_br','maximum',formula='Maximum signed horizontal hip–shoulder line separation FC→BR, with local unwrapping; not whole-clip P90.',refs=('game','review'))
    for ser,ko,en in [('elbow_flexion','투구팔 팔꿈치 굴곡','Throwing-elbow flexion'),('upper_arm_elevation','몸통 기준 상완 거상','Upper-arm elevation relative to thorax')]:
        for event in ['fc','br']:
            add(ser+'_'+event,ko+(' · 착지' if event=='fc' else ' · 릴리스'),en+' at '+event.upper(),'arm',ser,event,
              formula='180° − 3D shoulder–elbow–wrist included angle; 0° straight.' if ser=='elbow_flexion' else 'acos(−u_upper_arm·Z_thorax); 0° arm down, 90° horizontal. Elevation only; NOT isolated frontal shoulder abduction or axial rotation.',refs=('obp','report'),difference='Joint-center flexion / thorax-relative vector elevation. Upper-arm elevation is explicitly not substituted for Visual3D shoulder abduction.')
    add('shoulder_horizontal_abduction_fc','착지 시 상완 수평외전 · 기하각','Upper-arm geometric horizontal abduction at FC','arm','shoulder_horizontal_abduction',formula='atan2(−upper_arm_local_anterior, outward_lateral); 0° arm laterally outward, positive posterior. Upper-arm horizontal projection norm must exceed 0.2. Geometric angle, not full anatomical shoulder joint-coordinate decomposition.',refs=('obp','report'))
    for id,ko,en,reason in [
      ('shoulder_abduction_fc','착지 시 해부학적 어깨 외전','Anatomical shoulder abduction at FC','anatomical_shoulder_frame_unavailable'),
      ('shoulder_abduction_br','릴리스 시 해부학적 어깨 외전','Anatomical shoulder abduction at BR','anatomical_shoulder_frame_unavailable'),
      ('shoulder_external_rotation_fc','착지 시 어깨 외회전','Shoulder external rotation at FC','humeral_axial_rotation_unidentifiable'),
      ('shoulder_external_rotation_max','최대 어깨 외회전','Maximum shoulder external rotation','humeral_axial_rotation_unidentifiable'),
      ('shoulder_external_rotation_peak_pct','최대 어깨 외회전 시점','Time of maximum shoulder external rotation','humeral_axial_rotation_unidentifiable')]:
        add(id,ko,en,'arm',event='fc' if id.endswith('_fc') else 'br' if id.endswith('_br') else 'fc_br',unit='% interval' if id.endswith('_pct') else '°',blocked=reason,formula=en+' requires anatomical segment orientation/axial-rotation observability; never inferred silently from forearm direction.',refs=('obp','accuracy'),correspondence='research_concept_not_measured')
    for ser,ko,en in [('lead_knee','앞무릎 신전','Lead-knee extension'),('pelvis','골반 회전','Pelvis rotation'),('trunk','몸통 회전','Trunk rotation'),('elbow','팔꿈치 신전','Elbow extension'),('shoulder_internal','어깨 내회전','Shoulder internal rotation')]:
        for suffix,unit,desc in [('peak_velocity','°/s','Maximum physical angular velocity'),('peak_time_ms','ms from FC','Physical time of maximum angular velocity')]:
            add(ser+'_'+suffix,ko+(' 최대 각속도' if suffix=='peak_velocity' else ' 최대 각속도 시점'),desc+' — '+en,'timing',event='fc_br',unit=unit,blocked='humeral_axial_rotation_unidentifiable' if ser=='shoulder_internal' else 'physical_time_unverified',formula=desc+' within FC→BR. Extension = −d(flexion)/dt; rotation uses unwrapped signed angle. Requires real-time scale, adequate native image cadence, recorded filter and peak sensitivity; currently no fabricated deg/s or ms.',refs=('obp','sequence','knee'),correspondence='research_concept_not_measured')
    add('fc_br_time_ms','착지→릴리스 실제 시간','Physical FC→BR duration','timing',event='fc_br',unit='ms',blocked='physical_time_unverified',formula='1000×(physical BR timestamp − physical FC timestamp), requires verified playback-to-physical time mapping.',refs=('obp',),correspondence='research_concept_not_measured')
    add('pelvis_trunk_peak_gap_ms','골반 피크→몸통 피크 시간차','Pelvis-to-trunk peak time gap','timing',event='fc_br',unit='ms',blocked='physical_time_unverified',formula='t(max trunk rotation velocity) − t(max pelvis rotation velocity), positive pelvis first; overlapping peak uncertainty = simultaneous or unresolved.',refs=('obp','sequence'),correspondence='research_concept_not_measured')
    add('stride_length_cm','착지 시 보폭','Stride at image contact','space',series='stride_length_cm',event='fc',unit='cm',blocked='metric_scale_unverified',formula='Distance between ankle joint centers at FC × calibrated length scale. OBP same-frame ankle distance chosen; differs from PKH rear-ankle to FC lead-ankle displacement in Manzi et al.',refs=('obp','pelvis'),correspondence='research_concept_not_measured')
    add('stride_height_pct','신장 대비 착지 보폭','Stride / measured stature','space',series='stride_height_pct',event='fc',unit='% stature',blocked='measured_stature_unavailable',formula='100×calibrated same-frame 3D ankle-center stride at FC / independently measured stature. Model height is not measured player height.',refs=('obp','pelvis'),correspondence='research_concept_not_measured')
    add('lead_foot_lateral_pct','착지 시 앞발 좌우 위치 · 모델 다리길이 정규화','Lead-foot lateral offset / model leg length','space',event='fc',operation='foot_lateral',unit='% model leg',formula='PROJECT DERIVED: 100×(lead ankle − trail ankle)·pitch-frame Y × hand_sign / (native hip–knee + knee–ankle length). Positive glove-side. Same-frame model geometry; not cm or stature-normalized stride.',refs=('obp',),correspondence='project_derived')
    add('lead_foot_direction_fc','착지 시 앞발 방향','Lead-foot direction relative to pitch axis','space',event='fc',operation='foot_direction',formula='Signed horizontal angle from plate direction to heel→mean(big toe,small toe) native SMPL surface direction, throwing-side normalized; requires independently observed heel and toe support. Ankles alone are insufficient.',refs=('obp',),difference='SMPL surface toe/heel estimate, not measured foot markers. Pitch axis is an explicit image-supported hypothesis; axis sensitivity retained.')
    short={
      'lead_knee_flexion_fc':'앞발이 영상에서 지면에 닿는 시점의 3D 무릎 굴곡입니다. 0°가 펴진 상태입니다.',
      'lead_knee_flexion_br':'공이 손에서 분리되는 영상 구간의 3D 무릎 굴곡입니다. 0°가 펴진 상태입니다.',
      'lead_knee_net_extension':'착지 굴곡각에서 릴리스 굴곡각을 뺀 값입니다. 양수이면 착지 후 더 펴졌다는 뜻입니다.',
      'lead_knee_max_flexion':'착지–릴리스 사이 관측 가능한 원본 프레임에서 가장 크게 굽혀진 각도입니다.',
      'lead_knee_extra_flexion':'착지 이후의 최대 굴곡각에서 착지 굴곡각을 뺀 값입니다.',
      'lead_knee_extension_transition_pct':'프로젝트 파생 변수입니다. 최대 굴곡 이후 두 관측 이미지에서 1° 이상 지속 감소하기 시작한 재생 구간 비율입니다.',
      'thorax_pelvis_axial_fc':'모델 골반 좌표계에서 계산한 흉곽의 상대 축회전입니다. 양수는 골반이 먼저 열린 방향입니다. 어깨선 분리각과 다릅니다.',
      'thorax_pelvis_axial_max':'착지–릴리스 구간의 최대 부호 있는 흉곽–골반 상대 축회전입니다. 절댓값이나 클립 전체 P90이 아닙니다.',
      'thorax_pelvis_axial_peak_pct':'최대 상대 축회전이 나타난 재생 구간 비율입니다. 실제 밀리초나 물리적 시간 비율은 아닙니다.',
      'hip_shoulder_line_separation_fc':'수평면에 투영한 골반 관절 중심선과 어깨 관절 중심선의 부호 있는 각도입니다. 흉곽–골반 상대 회전과 별도입니다.',
      'hip_shoulder_line_separation_max':'착지–릴리스 구간에서 골반선–어깨선의 분리각이 가장 큰 값입니다. 양수는 골반 선행 방향입니다.',
      'lead_foot_lateral_pct':'프로젝트 파생 변수입니다. 착지 시 앞발목의 뒷발목 대비 좌우 위치를 모델 다리 길이로 나눕니다. 신장 대비 보폭이 아닙니다.',
      'lead_foot_direction_fc':'착지 시 뒤꿈치→발가락 방향과 투구 축의 수평 각도입니다. 원본 발가락·뒤꿈치 관측과 투구 축이 필요합니다.'}
    for m in metrics:
        id=m['id'];series=m.get('series');inputs=['Known throwing hand / original left-right identities','Original image confidence, interpolation/outlier flags','Native source-frame IDs and declared image event brackets']
        if series=='lead_knee_flexion' or id.startswith('lead_knee_'):
            joints='SMPL lead hip–knee–ankle: R throw [1,4,7], L throw [2,5,8]; original Halpe R [11,13,15], L [12,14,16]'
            inputs+=['Native 3D lead hip, knee and ankle centers; no humeral or thorax rotation required']
        elif series=='elbow_flexion' or id.startswith('elbow_'):
            joints='SMPL throwing shoulder–elbow–wrist: R [17,19,21], L [16,18,20]; Halpe R [6,8,10], L [5,7,9]'
            inputs+=['Native 3D throwing shoulder, elbow and wrist centers']
        elif series=='shoulder_line_separation':
            joints='SMPL hips [1,2], shoulders [16,17]; original Halpe [5,6,11,12]'
            inputs+=['3D hip/shoulder joint-center lines','Clip-fixed inferred up; sufficiently long horizontal projections']
        elif series in ['upper_arm_elevation','shoulder_horizontal_abduction']:
            joints='SMPL upper arm R [17,19], L [16,18]; thorax chain [0,3,6,9] and neutral-rest neck/shoulders [9,12,16,17]; Halpe torso [5,6,11,12] and throwing upper arm'
            inputs+=['3D shoulder/elbow centers','Cumulative SMPL thorax orientation and native-rest anatomical frame calibration']
        elif id.startswith('shoulder_'):
            joints='Independently observable anatomical thorax/scapula/humerus frames and humeral long-axis orientation; joint centers alone are insufficient'
            inputs+=['Anatomical shoulder calibration','Observation-supported humeral/scapular orientation; SMPL local twist output alone does not establish observability']
        elif id.startswith('stride_') or id=='lead_foot_lateral_pct':
            joints='SMPL ankle centers [7,8]; lead hip–knee–ankle for model-leg normalization where specified; Halpe ankle support [15,16]'
            inputs+=['Same-FC native 3D ankle centers']
            if id=='stride_length_cm':inputs+=['Independently verified model-unit-to-meters scale and scale evidence']
            if id=='stride_height_pct':inputs+=['Independently verified model-unit-to-meters scale','Independently measured player stature and evidence']
        elif id=='lead_foot_direction_fc':
            joints='Lead foot native heel/big-toe/small-toe surface markers: R throw [0,2,4], L throw [1,3,5]; Halpe R [20,22,24], L [21,23,25]'
            inputs+=['Native surface heel and both toe directions','Independently supported original heel/toe image observations']
        elif id=='fc_br_time_ms':
            joints='No joint-angle input required: independently supported image FC and BR events'
            inputs+=['Physical timestamps of FC and BR; not a wrist-speed surrogate']
        else:
            joints='SMPL pelvis root 0; cumulative thorax chain [0,3,6,9]; neutral-rest hips [1,2], spine/neck [9,12], shoulders [16,17]; original Halpe [5,6,11,12]'
            inputs+=['Native SMPL local rotations and joint-center neutral-rest offsets','Cumulative segment orientations in the clip-fixed inferred-up frame']
        if series in ['pelvis_rotation','trunk_rotation','trunk_forward_lean','trunk_lateral_lean'] or id in ['lead_foot_lateral_pct','lead_foot_direction_fc','pelvis_trunk_peak_gap_ms'] or id.startswith(('pelvis_peak','trunk_peak')):
            inputs+=['Independently supported plate-direction hypothesis and its uncertainty; unavailable axis blocks this metric only']
        if 'peak_velocity' in id or 'peak_time_ms' in id or id in ['fc_br_time_ms','pelvis_trunk_peak_gap_ms']:
            inputs+=['Verified playback-to-physical time factor and independent clock evidence','Original independent-image cadence; native filtering/peak sensitivity assessment']
        m['requiredJoints']=joints;m['requiredInputs']=inputs
        if m['operation'] in ['maximum','argmax'] or 'peak_' in m['id']:
            m['differenceFromSource']+=' The declared FC→BR window can exclude earlier delivery peaks; it is not a full-delivery maximum. Physical peak filtering is a native-image polynomial sensitivity analysis, not the laboratory 13.4/20 Hz Butterworth pipeline.'
        if 'lean' in m['id']:
            m['differenceFromSource']+=' Forward/lateral fields are separate global-plane inclinations of the model thorax superior axis, not Visual3D Cardan anterior/lateral angles.'
        if m['id'] in short:m['definition']['ko']=short[m['id']]
        elif m['id'].startswith('elbow_flexion'):m['definition']['ko']='투구팔 어깨–팔꿈치–손목 세 관절 중심으로 계산한 3D 굴곡각입니다. 0°가 펴진 상태입니다.'
        elif m['id'].startswith('upper_arm_elevation'):m['definition']['ko']='몸통 아래 방향에 대한 상완의 3D 각도입니다. 0°는 팔을 내린 상태, 90°는 수평입니다. 해부학적 어깨 외전·외회전과 구분합니다.'
        elif m['id']=='shoulder_horizontal_abduction_fc':m['definition']['ko']='몸통 수평면에서 옆으로 뻗은 상완을 기준으로 뒤쪽에 놓인 정도입니다. 상완 수평 투영이 너무 작으면 산출하지 않습니다.'
        elif m.get('blockedReason'):m['definition']['ko']=m['name']['ko']+'입니다. 필요한 실제 시간·좌표계·스케일 또는 상완 축회전 식별 요건이 없어 현재 산출하지 않습니다. 자세한 계산 정의와 차단 이유를 확인하세요.'
        elif m['id'].startswith('trunk_forward_lean'):m['definition']['ko']='투구 축과 전역 상방을 기준으로 분해한 몸통 전경입니다. 양수는 홈플레이트 쪽 기울기입니다. 단순 수직선 기울기와 다릅니다.'
        elif m['id'].startswith('trunk_lateral_lean'):m['definition']['ko']='전경·축회전과 분리한 몸통 측굴입니다. 좌우투를 통일하여 양수는 글러브 쪽 기울기입니다.'
        elif 'rotation' in m['id']:m['definition']['ko']='모델의 중립 자세로 보정한 분절 좌표계의 부호 있는 축회전입니다. 투구 축이 필요하며, 0°는 앞쪽이 홈플레이트를 향한 상태입니다.'
    return metrics
