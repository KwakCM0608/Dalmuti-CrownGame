CREATE TABLE `online_room_members` (
	`room_code` text NOT NULL,
	`player_id` text NOT NULL,
	`nickname` text NOT NULL,
	`nickname_key` text NOT NULL,
	`token_hash` text NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	`last_seen_at` integer NOT NULL,
	PRIMARY KEY(`room_code`, `player_id`),
	FOREIGN KEY (`room_code`) REFERENCES `online_rooms`(`code`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE UNIQUE INDEX `online_room_members_token_idx` ON `online_room_members` (`token_hash`);--> statement-breakpoint
CREATE UNIQUE INDEX `online_room_members_nickname_idx` ON `online_room_members` (`room_code`,`nickname_key`);--> statement-breakpoint
CREATE TABLE `online_rooms` (
	`code` text PRIMARY KEY NOT NULL,
	`state_json` text NOT NULL,
	`revision` integer DEFAULT 0 NOT NULL,
	`created_at` integer NOT NULL,
	`updated_at` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `online_rooms_code_idx` ON `online_rooms` (`code`);