#[derive(Clone, Copy)]
pub struct DraftSchedule {
    gamma: usize,
    maximum: usize,
    adaptive: bool,
}

impl DraftSchedule {
    pub fn fixed(gamma: usize) -> Self {
        assert!(gamma > 0, "gamma must be positive");
        Self {
            gamma,
            maximum: gamma,
            adaptive: false,
        }
    }

    pub fn adaptive(initial: usize, maximum: usize) -> Result<Self, &'static str> {
        if initial == 0 || initial > maximum {
            return Err("adaptive gamma must be positive and no greater than max gamma");
        }
        Ok(Self {
            gamma: initial,
            maximum,
            adaptive: true,
        })
    }

    pub(crate) fn gamma(&self) -> usize {
        self.gamma
    }

    pub(crate) fn observe(&mut self, accepted: usize, drafted: usize) {
        assert!(drafted > 0 && accepted <= drafted);
        // A short final round doesn't tell us how the full draft budget performs.
        if !self.adaptive || drafted != self.gamma {
            return;
        }
        if accepted == drafted {
            self.gamma = self.gamma.saturating_add(1).min(self.maximum);
        } else if accepted < drafted.div_ceil(2) {
            self.gamma = (self.gamma / 2).max(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DraftSchedule;

    #[test]
    fn feedback_increases_and_decreases_budget_within_bounds() {
        let mut schedule = DraftSchedule::adaptive(4, 5).unwrap();
        schedule.observe(4, 4);
        assert_eq!(schedule.gamma(), 5);
        schedule.observe(5, 5);
        assert_eq!(schedule.gamma(), 5);
        schedule.observe(1, 5);
        assert_eq!(schedule.gamma(), 2);
        schedule.observe(0, 2);
        assert_eq!(schedule.gamma(), 1);
        schedule.observe(0, 1);
        assert_eq!(schedule.gamma(), 1);
    }

    #[test]
    fn fixed_and_truncated_rounds_keep_the_budget() {
        let mut fixed = DraftSchedule::fixed(5);
        fixed.observe(0, 5);
        assert_eq!(fixed.gamma(), 5);
        let mut adaptive = DraftSchedule::adaptive(5, 8).unwrap();
        adaptive.observe(1, 1);
        assert_eq!(adaptive.gamma(), 5);
        assert!(DraftSchedule::adaptive(0, 8).is_err());
        assert!(DraftSchedule::adaptive(9, 8).is_err());
    }
}
