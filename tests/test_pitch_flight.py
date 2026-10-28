"""Physics and publication checks for derived Naver pitch-flight values."""
import gzip
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('flight_builder',ROOT/'scripts/build_pitch_flight.py')
flight=importlib.util.module_from_spec(spec);spec.loader.exec_module(flight)


def pitch(**changes):
    values=dict(x0_ft=0,y0_ft=50,z0_ft=6,vx0_ft_s=0,vy0_ft_s=-130,vz0_ft_s=0,
                ax_ft_s2=0,ay_ft_s2=0,az_ft_s2=-flight.G,tracking_status='derived_validated_x')
    values.update(changes)
    return values


class PitchFlightTests(unittest.TestCase):
    def test_gravity_only_has_zero_induced_break(self):
        r=flight.calculate(pitch())
        self.assertEqual(r['status'],'derived_pts')
        self.assertEqual(r['hb_cm'],0)
        self.assertEqual(r['ivb_cm'],0)
        duration=(50-17/12)/130
        expected=math.degrees(math.atan2(-flight.G*duration,130))
        self.assertAlmostEqual(r['vaa_deg'],expected,places=4)

    def test_drag_parallel_to_mean_velocity_is_removed(self):
        # Construct an exactly known mean velocity over the 40-ft interval.
        dt=(40-17/12)/130
        acceleration=(-.6,26,-flight.G+1)
        initial=[v-a*dt/2 for v,a in zip((3,-130,-5),acceleration)]
        r=flight.calculate(pitch(y0_ft=40,vx0_ft_s=initial[0],vy0_ft_s=initial[1],vz0_ft_s=initial[2],
                                ax_ft_s2=acceleration[0],ay_ft_s2=acceleration[1],az_ft_s2=acceleration[2]))
        self.assertEqual(r['hb_cm'],0)
        self.assertEqual(r['ivb_cm'],0)

    def test_horizontal_reflection_changes_only_hb_sign(self):
        a=flight.calculate(pitch(vx0_ft_s=3,ax_ft_s2=-12,az_ft_s2=-12,ay_ft_s2=25))
        b=flight.calculate(pitch(vx0_ft_s=-3,ax_ft_s2=12,az_ft_s2=-12,ay_ft_s2=25))
        self.assertAlmostEqual(a['hb_cm'],-b['hb_cm'])
        self.assertEqual(a['ivb_cm'],b['ivb_cm'])
        self.assertEqual(a['vaa_deg'],b['vaa_deg'])

    def test_missing_and_impossible_fits_stay_null(self):
        for row in [pitch(tracking_status='missing_pts_option'),pitch(vy0_ft_s=130),pitch(ax_ft_s2='nan'),pitch(y0_ft=30)]:
            r=flight.calculate(row)
            self.assertNotEqual(r['status'],'derived_pts')
            self.assertIsNone(r['hb_cm']);self.assertIsNone(r['ivb_cm']);self.assertIsNone(r['vaa_deg'])

    def test_published_pitch_ids_and_missing_counts(self):
        raw=(ROOT/'data/portal/pitch_flight.json').read_bytes()
        self.assertEqual(gzip.decompress((ROOT/'data/portal/pitch_flight.json.gz').read_bytes()),raw)
        d=json.loads(raw)
        self.assertEqual(d['recordCount'],9754)
        self.assertEqual(d['statusCounts'],{'derived_pts':9747,'no_tracking':7})
        ids={r['id'] for path in (ROOT/'data/portal/players').glob('*.json') for r in json.loads(path.read_text())['records']}
        self.assertEqual(ids,set(d['rows']))
        for r in d['rows'].values():
            if r['status']=='derived_pts':
                self.assertTrue(all(math.isfinite(r[k]) for k in ['hb_cm','ivb_cm','vaa_deg']))
            else:self.assertIsNone(r['hb_cm'])


if __name__=='__main__':unittest.main()
