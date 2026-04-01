from __future__ import annotations

from typing import Iterable

import numpy as np
import pygame


class MaternalChildHealthRenderer:
    def __init__(self, width: int = 1100, height: int = 700, fps: int = 10) -> None:
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maternal & Child Health Mission Simulator")
        self.clock = pygame.time.Clock()

        self.bg = (16, 22, 33)
        self.bg_top = (20, 30, 48)
        self.bg_bottom = (10, 16, 28)
        self.shadow = (8, 12, 20)
        self.panel = (29, 37, 53)
        self.panel_alt = (24, 33, 49)
        self.text = (236, 240, 245)
        self.good = (56, 176, 111)
        self.warn = (230, 168, 52)
        self.bad = (220, 76, 70)
        self.accent = (70, 150, 250)

        self.font_title = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 15)

        self.labels = [
            "Maternal Risk",
            "Neonatal Risk",
            "Immunization Gap",
            "Nutrition Deficit",
            "Referral Backlog",
            "Supply Stress",
            "Community Trust",
            "Weather Severity",
            "Staff Fatigue",
            "Budget",
            "Day Progress",
            "Recent Shock",
        ]

    def set_fps(self, fps: int) -> None:
        self.fps = max(1, int(fps))

    def _bar_color(self, value: float, reverse: bool = False):
        if reverse:
            value = 1.0 - value
        if value < 0.33:
            return self.good
        if value < 0.67:
            return self.warn
        return self.bad

    def _draw_gradient_background(self, canvas: pygame.Surface) -> None:
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(self.bg_top[0] * (1.0 - t) + self.bg_bottom[0] * t)
            g = int(self.bg_top[1] * (1.0 - t) + self.bg_bottom[1] * t)
            b = int(self.bg_top[2] * (1.0 - t) + self.bg_bottom[2] * t)
            pygame.draw.line(canvas, (r, g, b), (0, y), (self.width, y))

    def _draw_card(self, canvas: pygame.Surface, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
        shadow_rect = rect.move(3, 4)
        pygame.draw.rect(canvas, self.shadow, shadow_rect, border_radius=14)
        pygame.draw.rect(canvas, color, rect, border_radius=14)

    def _draw_header(
        self,
        canvas: pygame.Surface,
        day: int,
        max_days: int,
        total_reward: float,
        last_action: str = "-",
        last_reward: float = 0.0,
    ):
        title = self.font_title.render("Mission: Maternal & Child Health Operations", True, self.text)
        canvas.blit(title, (24, 16))

        line = self.font_body.render(
            f"Day {day:03d}/{max_days}    Cumulative Reward: {total_reward: .2f}",
            True,
            self.accent,
        )
        canvas.blit(line, (26, 55))

        action_color = self.good if last_reward >= 0 else self.warn
        action_line = self.font_small.render(
            f"Last Action: {last_action}   |   Step Reward: {last_reward:+.3f}",
            True,
            action_color,
        )
        canvas.blit(action_line, (26, 78))

    def _draw_trend_panel(self, canvas: pygame.Surface, risk_trend: list[float], reward_trend: list[float]):
        panel = pygame.Rect(24, 630, 1051, 55)
        self._draw_card(canvas, panel, self.panel_alt)

        label = self.font_small.render("Mission trend (recent 60 steps): risk burden + reward", True, self.text)
        canvas.blit(label, (36, 637))

        if not risk_trend and not reward_trend:
            return

        plot_rect = pygame.Rect(330, 638, 730, 40)
        pygame.draw.rect(canvas, (18, 25, 39), plot_rect, border_radius=8)

        def _norm(vals: list[float]) -> list[float]:
            if not vals:
                return []
            vmin = min(vals)
            vmax = max(vals)
            if abs(vmax - vmin) < 1e-9:
                return [0.5 for _ in vals]
            return [(v - vmin) / (vmax - vmin) for v in vals]

        def _draw_line(vals: list[float], color: tuple[int, int, int]):
            if len(vals) < 2:
                return
            nvals = _norm(vals)
            points = []
            for i, v in enumerate(nvals):
                x = plot_rect.x + int(i * (plot_rect.width - 4) / (len(nvals) - 1)) + 2
                y = plot_rect.bottom - int(v * (plot_rect.height - 4)) - 2
                points.append((x, y))
            pygame.draw.lines(canvas, color, False, points, 2)

        _draw_line(risk_trend[-60:], self.bad)
        _draw_line(reward_trend[-60:], self.accent)

        legend_1 = self.font_small.render("Risk burden", True, self.bad)
        legend_2 = self.font_small.render("Reward", True, self.accent)
        canvas.blit(legend_1, (190, 658))
        canvas.blit(legend_2, (265, 658))

    def _draw_status_bars(self, canvas: pygame.Surface, state: np.ndarray):
        panel = pygame.Rect(24, 95, 530, 575)
        self._draw_card(canvas, panel, self.panel)

        x0 = 44
        y0 = 120
        bar_w = 320
        bar_h = 20
        spacing = 43

        for idx, (label, value) in enumerate(zip(self.labels, state)):
            y = y0 + idx * spacing
            txt = self.font_body.render(label, True, self.text)
            canvas.blit(txt, (x0, y))

            frame = pygame.Rect(x0 + 165, y + 2, bar_w, bar_h)
            pygame.draw.rect(canvas, (55, 64, 82), frame, border_radius=8)
            fill = pygame.Rect(x0 + 165, y + 2, int(bar_w * float(value)), bar_h)

            reverse = label == "Community Trust" or label == "Budget"
            pygame.draw.rect(canvas, self._bar_color(float(value), reverse=reverse), fill, border_radius=8)

            val_text = self.font_body.render(f"{float(value):.2f}", True, self.text)
            canvas.blit(val_text, (x0 + 500, y))

    def _draw_mission_map(self, canvas: pygame.Surface, state: np.ndarray):
        panel = pygame.Rect(575, 95, 500, 575)
        self._draw_card(canvas, panel, self.panel)

        map_rect = pygame.Rect(600, 125, 450, 430)
        pygame.draw.rect(canvas, (20, 27, 40), map_rect, border_radius=12)

        # Draw district zones
        zones = [
            pygame.Rect(620, 145, 130, 120),
            pygame.Rect(765, 145, 130, 120),
            pygame.Rect(910, 145, 120, 120),
            pygame.Rect(620, 290, 180, 110),
            pygame.Rect(815, 290, 215, 110),
            pygame.Rect(620, 415, 410, 120),
        ]

        risk = float((state[0] + state[1]) / 2.0)
        supply_stress = float(state[5])
        fatigue = float(state[8])

        zone_labels = [
            "Peri-urban",
            "Rural North",
            "Rural East",
            "Market District",
            "District Hospital",
            "River Settlements",
        ]

        for zone, label in zip(zones, zone_labels):
            zone_offset = (zone.x % 17) / 100.0
            zone_intensity = min(1.0, max(0.0, 0.4 * risk + 0.35 * supply_stress + 0.25 * fatigue + zone_offset - 0.08))
            color = self._bar_color(zone_intensity)
            pygame.draw.rect(canvas, color, zone, border_radius=10)

            txt = self.font_body.render(label, True, (10, 15, 22))
            canvas.blit(txt, (zone.x + 8, zone.y + 8))

        # Draw mission legend
        legend_y = 570
        legend_items = [
            (self.good, "Stable"),
            (self.warn, "Needs Attention"),
            (self.bad, "High Priority"),
        ]
        x = 610
        for color, text in legend_items:
            pygame.draw.circle(canvas, color, (x, legend_y), 8)
            txt = self.font_body.render(text, True, self.text)
            canvas.blit(txt, (x + 14, legend_y - 10))
            x += 150

    def draw(
        self,
        state: Iterable[float],
        day: int,
        max_days: int,
        total_reward: float,
        last_action: str = "-",
        last_reward: float = 0.0,
        risk_trend: list[float] | None = None,
        reward_trend: list[float] | None = None,
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        state_arr = np.array(list(state), dtype=np.float32)
        canvas = pygame.Surface((self.width, self.height))
        self._draw_gradient_background(canvas)

        self._draw_header(canvas, day, max_days, total_reward, last_action=last_action, last_reward=last_reward)
        self._draw_status_bars(canvas, state_arr)
        self._draw_mission_map(canvas, state_arr)
        self._draw_trend_panel(canvas, risk_trend or [], reward_trend or [])

        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2))

    def show(self, frame: np.ndarray):
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
